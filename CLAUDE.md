# EGFR Driver Mutation Classifier — SlideFlow + DINOv2

Binary MIL classification of EGFR driver mutations from TCGA-LUAD whole-slide images.
Label: `egfr_driver` (1 = FDA/NCCN driver mutation, 0 = EGFR wild-type or non-driver VUS).

---

## Environment setup

### Python virtual environment

```bash
# Activate the project venv (Python 3.12 — Python 3.13 is incompatible with SlideFlow)
source .venv/bin/activate
```

**Why Python 3.12:** SlideFlow uses `imghdr`, which was removed in Python 3.13.

**Key packages installed:**
- `slideflow==3.0.2` (installed `--no-deps` to avoid pyrfr/smac build failures)
- `torch==2.6.0+cu124` (CUDA 12.4, for NVIDIA L4 GPU)
- `torchvision==0.21.0+cu124`
- `fastai` (required by SlideFlow MIL trainer — installed separately)
- `timm`, `huggingface_hub` (for foundation model loading)

**Venv patches applied (SlideFlow bugs):**
- `.venv/lib/python3.12/site-packages/slideflow/io/torch/img_utils.py`:
  `np.fromstring(image, dtype=np.uint8)` → `np.frombuffer(image, dtype=np.uint8).copy()`
  (NumPy 2.0 removed binary mode of `fromstring`; `.copy()` makes array writable for PyTorch)
- `.venv/lib/python3.12/site-packages/slideflow/mil/train/_fastai.py` line 140:
  `pd.value_counts(targets[train_idx])` → `pd.Series(targets[train_idx]).value_counts()`
  (pandas 2.x removed top-level `pd.value_counts`)

### GCS slides — gcsfuse mount

Slides are stored in `gs://wsi_bucket53/`. SlideFlow uses Python glob for slide discovery
(doesn't support `gs://` URIs), so the bucket is mounted via gcsfuse:

```bash
# Mount (if not already mounted)
gcsfuse --implicit-dirs wsi_bucket53 /home/satya/Github/slideflow-egfr/slides/wsi_bucket53

# Verify
ls slides/wsi_bucket53/TCGA_LUAD_SVS/ | head
```

`project/datasets.json` uses the local mount paths (already configured).

---

## What we have set up

### Data

**GCS slide sources (3 buckets):**

| Bucket folder | Local mount path | Slides |
|---|---|---|
| `gs://wsi_bucket53/TCGA_LUAD_SVS/` | `slides/wsi_bucket53/TCGA_LUAD_SVS/` | 1,000 |
| `gs://wsi_bucket53/egfr_exon19_luad/` | `slides/wsi_bucket53/egfr_exon19_luad/` | 80 |
| `gs://wsi_bucket53/EGFR_SVS/` | `slides/wsi_bucket53/EGFR_SVS/` | 816 |

**`annotations.csv` — 1,060 deduplicated TCGA slides:**

| Column | Description |
|---|---|
| `patient` | TCGA 12-char barcode (`TCGA-XX-XXXX`) |
| `slide` | SVS filename without extension (SlideFlow key) |
| `gcs_path` | Full `gs://` URI |
| `source_bucket` | Which bucket folder the file lives in |
| `sample_type_code` | `01` = Primary Tumor, `11` = Solid Normal |
| `slide_type` | `DX` (diagnostic) / `TS` (top section) / `BS` (bottom section) |
| `cancer_type` | `LUAD` or `LUSC` |
| `egfr_driver` | **Training label** — 1 = driver, 0 = WT/non-driver |
| `egfr_mutated` | 1 = any EGFR mutation (driver + VUS/other) |
| `protein_change` | Specific EGFR protein change(s) |
| `driver_category` | Drug class annotation (e.g. "L858R Exon 21 – Classical") |
| `egfr_class` | Driver / VUS-Missense / Nonsense / Frameshift / Splice / WT |

**LUAD training set (default — LUSC excluded):**

```
Primary tumor slides:  837  (417 patients)
  EGFR Driver (1):     138 slides  /  62 patients
  WT / non-driver (0): 699 slides  / 355 patients
  Ratio WT:Driver:     5.1:1
  pos_weight for BCE:  5.07
```

Slide type breakdown (LUAD primary tumor):

```
             WT   Driver
DX           298    58      ← preferred for training (diagnostic H&E)
TS           209    44
BS           192    36
```

**LUSC slides** (42 slides, 16 patients) are present in `annotations.csv` but
excluded from training by default. Pass `--include-lusc` to `train_mil.py` to include them.
Four LUSC patients have genuine driver mutations (L858R, L861Q, E746_A750del).

### Mutation labels — how they were derived

Mutations were queried from **cBioPortal** across 8 TCGA studies:
`luad_tcga`, `luad_tcga_pan_can_atlas_2018`, `luad_tcga_gdc`, `luad_tcga_pub`,
`lusc_tcga`, `lusc_tcga_pan_can_atlas_2018`, `lusc_tcga_gdc`, `lusc_tcga_pub`

**Driver classification (FDA/NCCN guidelines):**
- Exon 19 deletions — codon range 728–761 (classical: E746_A750del etc.)
- Exon 19 complex delins — same codon range (e.g. E746_S752delinsV)
- L858R — Exon 21
- T790M — Exon 20 resistance mutation
- G719X — Exon 18 uncommon
- L861Q — Exon 21 uncommon
- S768I — Exon 20 uncommon
- E709X — Exon 18 uncommon
- E709_T710delinsD — Exon 18 complex delins (NCCN)
- Exon 20 insertions/duplications — codon ≥ 762 (amivantamab/mobocertinib)

Non-driver EGFR mutations (VUS, splice, nonsense, frameshift) get `egfr_driver=0`.

New slides (76 TCGA) were discovered via the **GDC API** and streamed directly
to GCS with zero local disk usage. See `../tcga-luad-egfr/` repo for the
discovery and download scripts.

### SlideFlow project

- **Location:** `./project/`
- **Settings:** `./project/settings.json` — project name `TCGA-LUAD-EGFR`
- **Datasets:** `./project/datasets.json` — 3 gcsfuse-mounted slide sources
- **Annotations:** `./annotations.csv` — loaded by all scripts automatically
- **SlideFlow version:** 3.0.2

---

## Feature extraction

**Extractor: DINOv2 ViT-L/14** (Meta AI, Apache-2.0)
- Custom extractor: `dinov2_extractor.py`
- 1024-dim CLS token embeddings, frozen (no fine-tuning)
- Weights loaded via `torch.hub` from Facebook CDN (no token required)
- Input: 256px tiles resized to 224px (ViT-L/14 requires multiples of 14)
- Normalization: ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
- Output: `.pt` (PyTorch tensor) + `.index.npz` per slide in `features/dinov2_vitl14/`

**Note:** H-optimus-0 (Bioptimus, 1536-dim) was considered but abandoned due to
slow HuggingFace download speed on this VM. Can retry with `hoptimus_extractor.py`
if HF access improves (likely the biggest single AUC improvement available).

**Status: COMPLETE — 872 bags extracted (879 slides attempted, 7 failed due to missing MPP metadata).**

```bash
# Re-run extraction (incremental — already-extracted slides are skipped)
source .venv/bin/activate
nohup python -u extract_features.py > logs/extraction_full.log 2>&1 &

# Dry run (check plan without running)
python extract_features.py --dry-run

# Pilot run (balanced subset)
python extract_features.py --sample 100
```

**Disk management:** `extract_features.py` processes slides in batches of 150
(configurable via `--batch-slides N`). TFRecords (~384 MB/slide) are deleted after
each batch once the `.pt` bag is confirmed saved. This keeps disk usage bounded
to ~60 GB headroom rather than requiring ~245 GB for all slides at once.

---

## MIL Training

```bash
source .venv/bin/activate
nohup python -u train_mil.py > logs/train_mil.log 2>&1 &

# Key options:
#   --model        attention_mil | transmil | clam_sb | clam_mb  (default: attention_mil)
#   --folds        3    stratified k-fold CV (default: 3)
#   --epochs       20
#   --lr           1e-4
#   --dx-only           use DX (diagnostic) slides only — highest quality H&E
#   --include-lusc      add LUSC slides to training set
```

**Architecture:**
- ABMIL (Ilse et al. 2018) — attention-based aggregation
- `n_in=1024` (DINOv2 feature dim), `n_out=2` (binary classification)
- Patient-level aggregation: all slides per patient merged into one bag
- Weighted cross-entropy (`weighted_loss=True`) for 5:1 WT:Driver imbalance
- fastai one-cycle LR schedule, AdamW `lr=1e-4`, `wd=1e-5`, `bag_size=512`

**CV strategy — why 3-fold, not 5-fold:**
With only 62 driver patients, 5-fold CV leaves ~12 driver patients per val fold —
too few for stable AUC estimates (pilot showed 0.50–0.785 variance across folds).
3-fold gives ~21 driver patients per val fold. Standard in TCGA-scale pathology ML papers.

**Out-of-fold (OOF) aggregation:**
Per-fold `predictions.parquet` files are pooled into `mil/oof_predictions.parquet`
after training. A single AUC/AUPRC is computed over all patients (each appears in
validation exactly once). This is the number to report — not the mean of per-fold AUCs.

**Models saved to:** `./mil/fold{1..3}/`

**Recommended model order:**
1. `attention_mil` — ABMIL, standard baseline, fastest
2. `clam_sb` — CLAM single-branch, good for interpretability
3. `transmil` — transformer-based, better with large bag sizes

---

## Evaluation

```bash
source .venv/bin/activate
python evaluate_mil.py [--outdir ./mil] [--model-tag attention_mil-egfr_driver]
```

Reads `mil/oof_predictions.parquet` (written by `train_mil.py`) and outputs:
- `mil/roc_curve.png` — OOF aggregate ROC (black) + per-fold curves (colour)
- `mil/prc_curve.png` — OOF aggregate PRC + random baseline
- `mil/summary.json` — AUC, AUPRC, F1, sensitivity, specificity + per-fold breakdown

**Full cohort results — Run 1 (872 bags, 307 patients, 3-fold CV, ABMIL + DINOv2):**

| Fold | Val patients | Driver | AUC | AUPRC |
|------|-------------|--------|-----|-------|
| 1 | 139 | 20 | 0.500 | 0.331 |
| 2 | 139 | 21 | 0.733 | 0.355 |
| 3 | 139 | 21 | 0.610 | 0.212 |
| **OOF** | **307** | **62** | **0.580** | **0.256** |

OOF AUC 0.580 is above chance but below target (0.72). AUPRC of 0.256 vs random
baseline of ~0.165 confirms signal is present. Next steps: try `--dx-only` (removes
TS/BS noise) and/or H-optimus-0 features (pathology-native 1536-dim embeddings).

**Pilot results (198 bags, 137 patients, 5-fold CV — for reference only):**

| **OOF** | **137** | **55** | **0.578** | **0.543** |

---

## Roadmap

### Completed
- [x] Pilot feature extraction (198 bags, balanced 100+100)
- [x] Pilot training (5-fold, ABMIL, DINOv2) — OOF AUC 0.578
- [x] Full feature extraction — 872 bags, 879 slides attempted, 7 failed (missing MPP)
- [x] Full cohort training (3-fold, ABMIL, DINOv2) — OOF AUC 0.580, AUPRC 0.256
- [x] `evaluate_mil.py` — OOF aggregation, ROC + PRC plots, summary JSON
- [x] `setup.sh` — reproducible environment setup for fresh VMs

### Next (AUC 0.58 < target 0.72 — try these in order)

1. **`--dx-only`** — train on DX (diagnostic) slides only
   - 298 WT + 58 driver = 356 slides, removes noisier TS/BS sections
   - `nohup python -u train_mil.py --dx-only > logs/train_mil_dxonly.log 2>&1 &`

2. **`clam_sb`** — CLAM single-branch aggregator
   - Adds instance-level clustering loss, often better than ABMIL with small cohorts
   - `nohup python -u train_mil.py --model clam_sb > logs/train_mil_clam.log 2>&1 &`

3. **H-optimus-0 features** — pathology-native 1536-dim vs ImageNet DINOv2 1024-dim
   - Likely biggest single gain; requires HF token + good download speed
   - `hoptimus_extractor.py` is already written; re-run `extract_features.py` with it
   - `HF_TOKEN=hf_... python extract_features.py` (set extractor to `hoptimus`)

4. **`--include-lusc`** — adds 4 more driver patients (marginal)

### If AUC > 0.72
- Attention heatmaps — high-attention tiles should localise to tumour nuclei /
  gland architecture in driver cases (`mil/fold*/attention/*.npz` already saved)
- Try `transmil` — transformer-based aggregation, captures tile-to-tile context
- External validation — NLST or CPTAC-LUAD if accessible

---

## Related repo

`/home/satya/Github/tcga-luad-egfr/` — mutation discovery pipeline:

| File | Purpose |
|---|---|
| `check_egfr_mutations.py` | Query cBioPortal for EGFR mutations in GCS bucket slides |
| `egfr_gdc_discovery.py` | GDC-wide search for LUAD/LUSC slides with EGFR mutations |
| `download_egfr_slides.py` | Stream 76 new slides GDC → `gs://wsi_bucket53/EGFR_SVS/` |
| `egfr_gdc_discovery.xlsx` | Full mutation + slide inventory (6 sheets) |
| `egfr_mutation_status.xlsx` | Original TCGA_LUAD_SVS mutation labels (3 sheets) |
