#!/usr/bin/env python3
"""
extract_features.py
-------------------
Tile and feature-extract all primary-tumor slides using Hibou-L (ViT-L/14).

Features are saved as per-slide HDF5 bags in:
    ./features/<extractor_name>/<slide_name>.h5

Each .h5 contains:
    features  – (N, D) float32  feature vectors  (D=1024 for Hibou-L)
    coords    – (N, 2) int32    (x, y) top-left tile coords at full resolution

Usage:
    python extract_features.py [--workers 4] [--batch-size 32] [--dry-run]

Notes:
    - Slides are streamed directly from GCS via gsutil
    - Tiles are extracted at 256 px / 128 µm (≈ 20× equivalent)
    - Hibou-L weights are downloaded automatically from HuggingFace
    - Already-extracted slides are skipped (incremental)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import slideflow as sf
from slideflow.model import build_feature_extractor

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_DIR     = Path(__file__).parent
PROJECT_DIR  = REPO_DIR / "project"
ANN_CSV      = REPO_DIR / "annotations.csv"
FEATURES_DIR = REPO_DIR / "features"

TILE_PX  = 256
TILE_UM  = 128
QC       = "otsu"          # tissue segmentation: 'otsu', 'gaussian', or None
EXTRACTOR = "hibou_l"      # registered name in slideflow (via slideflow-gpl or HF)

SLIDE_DIRS = [
    "gs://wsi_bucket53/TCGA_LUAD_SVS",
    "gs://wsi_bucket53/egfr_exon19_luad",
    "gs://wsi_bucket53/EGFR_SVS",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_training_slides(ann_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ann_csv)
    # Only primary tumor slides with known label
    df = df[(df["sample_type_code"] == "01") & (df["egfr_driver"].isin([0, 1]))]
    print(f"Training slides: {len(df)} ({df['egfr_driver'].sum()} driver, "
          f"{(df['egfr_driver'] == 0).sum()} WT)")
    return df


def already_extracted(slide_name: str, features_dir: Path, extractor: str) -> bool:
    h5 = features_dir / extractor / f"{slide_name}.h5"
    return h5.exists() and h5.stat().st_size > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers",    type=int, default=4,
                    help="Tile-extraction worker threads per slide")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="GPU batch size for Hibou-L forward pass")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print plan without extracting")
    args = ap.parse_args()

    df = load_training_slides(ANN_CSV)

    # Open (or create) the SlideFlow project
    P = sf.Project(
        root=str(PROJECT_DIR),
        annotations=str(ANN_CSV),
        slides=SLIDE_DIRS,
    )

    # Build dataset filtered to primary tumor slides with known label
    dataset = P.dataset(
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        filters={
            "sample_type_code": ["01"],
            "egfr_driver": [0, 1],
        },
    )

    print(f"\nDataset: {dataset.num_slides()} slides")

    if args.dry_run:
        print("Dry-run mode: exiting without extraction.")
        return

    # Build feature extractor (downloads model weights on first run)
    print(f"\nBuilding feature extractor: {EXTRACTOR} …")
    extractor = build_feature_extractor(
        EXTRACTOR,
        tile_px=TILE_PX,
    )

    # Extract features (incremental — already-done slides are skipped)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting features → {FEATURES_DIR / EXTRACTOR} …")

    dataset.extract_feature_bags(
        extractor,
        outdir=str(FEATURES_DIR / EXTRACTOR),
        qc=QC,
        num_threads=args.workers,
        batch_size=args.batch_size,
    )

    print("\nFeature extraction complete.")
    bags = list((FEATURES_DIR / EXTRACTOR).glob("*.h5"))
    print(f"  Bags saved: {len(bags)}")


if __name__ == "__main__":
    main()
