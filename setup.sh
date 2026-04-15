#!/usr/bin/env bash
# =============================================================================
# setup.sh — reproducible environment for EGFR Driver Mutation Classifier
#
# Requirements:
#   - miniconda/conda installed (for Python 3.12)
#   - NVIDIA GPU with CUDA 12.4 drivers
#   - Ubuntu/Debian (for apt deps)
#   - GCS bucket access (for gcsfuse slide mount)
#
# Usage:
#   bash setup.sh
#   source .venv/bin/activate
#   bash setup.sh --mount   # also mounts GCS bucket
# =============================================================================

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
MOUNT_FLAG="${1:-}"

echo "============================================="
echo " EGFR Classifier — Environment Setup"
echo " Repo: $REPO_DIR"
echo "============================================="

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    libvips libvips-dev \
    libgl1 libglib2.0-0 \
    build-essential \
    curl

# gcsfuse
if ! command -v gcsfuse &> /dev/null; then
    echo "  Installing gcsfuse..."
    export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
        | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo apt-key add -
    sudo apt-get update -qq
    sudo apt-get install -y -qq gcsfuse
else
    echo "  gcsfuse already installed: $(gcsfuse --version 2>&1 | head -1)"
fi

# -----------------------------------------------------------------------------
# 2. Python 3.12 via conda
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Locating Python 3.12..."

# Try conda pkg cache first (fast, no download)
PY312=""
for candidate in \
    "$(conda run -n base python3.12 -c 'import sys; print(sys.executable)' 2>/dev/null || true)" \
    "$(find "$HOME/miniconda3/pkgs" "$HOME/anaconda3/pkgs" /opt/conda/pkgs \
        -name "python3.12" -type f 2>/dev/null | head -1 || true)"; do
    if [[ -x "$candidate" ]]; then
        PY312="$candidate"
        break
    fi
done

# Fallback: create a temporary conda env to get Python 3.12
if [[ -z "$PY312" ]]; then
    echo "  Python 3.12 not found in conda cache — creating temp conda env..."
    conda create -y -n _py312_tmp python=3.12 -q
    PY312="$(conda run -n _py312_tmp python -c 'import sys; print(sys.executable)')"
fi

echo "  Using Python: $PY312 ($($PY312 --version))"

# -----------------------------------------------------------------------------
# 3. Create virtual environment
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Creating virtual environment at $VENV_DIR..."

if [[ -d "$VENV_DIR" ]]; then
    echo "  .venv already exists — skipping creation (delete it to rebuild)"
else
    "$PY312" -m venv "$VENV_DIR"
    echo "  Created."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# -----------------------------------------------------------------------------
# 4. Install packages
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing Python packages..."

# setuptools pin must come first (slideflow needs pkg_resources)
pip install "setuptools==70.2.0" -q

# PyTorch with CUDA 12.4 — must use custom index URL
echo "  Installing PyTorch 2.6.0+cu124..."
pip install \
    "torch==2.6.0+cu124" \
    "torchvision==0.21.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124 \
    -q

# SlideFlow must be installed --no-deps (pyrfr/smac fail to build and aren't needed)
echo "  Installing SlideFlow 3.0.2 (--no-deps)..."
pip install "slideflow==3.0.2" --no-deps -q

# Everything else
echo "  Installing remaining dependencies..."
pip install -r "$REPO_DIR/requirements.txt" -q

echo "  Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; \
           print(f'  CUDA OK — {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}')"

# -----------------------------------------------------------------------------
# 5. Apply SlideFlow source patches
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Applying SlideFlow patches..."

SF_PATH="$VENV_DIR/lib/python3.12/site-packages/slideflow"

# Patch 1: np.fromstring → np.frombuffer (NumPy 2.0 removed binary mode)
#          + .copy() to make array writable for PyTorch
IMG_UTILS="$SF_PATH/io/torch/img_utils.py"
if grep -q "np.fromstring" "$IMG_UTILS" 2>/dev/null; then
    sed -i 's/np\.fromstring(image, dtype=np\.uint8)/np.frombuffer(image, dtype=np.uint8).copy()/g' "$IMG_UTILS"
    echo "  ✓ img_utils.py: np.fromstring → np.frombuffer().copy()"
elif grep -q "np.frombuffer(image, dtype=np.uint8).copy()" "$IMG_UTILS" 2>/dev/null; then
    echo "  ✓ img_utils.py: already patched"
else
    echo "  ⚠ img_utils.py: could not find target line — check manually"
fi

# Patch 2: pd.value_counts → pd.Series.value_counts (pandas 2.x)
FASTAI_TRAIN="$SF_PATH/mil/train/_fastai.py"
if grep -q "pd\.value_counts(" "$FASTAI_TRAIN" 2>/dev/null; then
    sed -i 's/pd\.value_counts(\(.*\))/pd.Series(\1).value_counts()/g' "$FASTAI_TRAIN"
    echo "  ✓ _fastai.py: pd.value_counts → pd.Series.value_counts()"
elif grep -q "pd.Series.*value_counts" "$FASTAI_TRAIN" 2>/dev/null; then
    echo "  ✓ _fastai.py: already patched"
else
    echo "  ⚠ _fastai.py: could not find target line — check manually"
fi

# -----------------------------------------------------------------------------
# 6. GCS bucket mount (optional)
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] GCS slide mount..."

MOUNT_DIR="$REPO_DIR/slides/wsi_bucket53"
BUCKET="wsi_bucket53"

if [[ "$MOUNT_FLAG" == "--mount" ]]; then
    mkdir -p "$MOUNT_DIR"
    if mountpoint -q "$MOUNT_DIR"; then
        echo "  Already mounted at $MOUNT_DIR"
    else
        echo "  Mounting gs://$BUCKET → $MOUNT_DIR"
        gcsfuse --implicit-dirs "$BUCKET" "$MOUNT_DIR"
        echo "  ✓ Mounted. Verifying..."
        ls "$MOUNT_DIR/" | head -3
    fi
else
    echo "  Skipped (run with --mount to also mount GCS bucket)"
    echo "  Manual mount: gcsfuse --implicit-dirs $BUCKET $MOUNT_DIR"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup complete!"
echo "============================================="
echo ""
echo " Activate:  source .venv/bin/activate"
echo " Mount GCS: bash setup.sh --mount"
echo " Extract:   python extract_features.py --dry-run"
echo " Train:     python train_mil.py"
echo " Evaluate:  python evaluate_mil.py"
echo ""
