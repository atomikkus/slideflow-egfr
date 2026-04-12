# EGFR Driver Mutation Classifier — SlideFlow + Hibou-L

Binary MIL classification of EGFR driver mutations from TCGA-LUAD whole-slide images.
Label: `egfr_driver` (1 = FDA/NCCN driver mutation, 0 = EGFR wild-type or non-driver VUS).

---

## What we have set up

### Data

**GCS slide sources (3 buckets):**

| Bucket folder | Contents | Slides |
|---|---|---|
| `gs://wsi_bucket53/TCGA_LUAD_SVS/` | Original TCGA-LUAD cohort | 1,000 |
| `gs://wsi_bucket53/egfr_exon19_luad/` | Pre-curated EGFR exon-19 LUAD cases | 80 |
| `gs://wsi_bucket53/EGFR_SVS/` | New TCGA slides downloaded from GDC this session (76 TCGA + 740 clinical non-TCGA) | 816 |

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
- **Datasets:** `./project/datasets.json` — 3 GCS slide sources registered
- **Annotations:** `./annotations.csv` — loaded by all scripts automatically
- **SlideFlow version:** 3.0.2

---

## What still needs to be done

### Step 1 — Feature extraction (GPU required)

```bash
python extract_features.py
# Options:
#   --workers 4       tile-extraction threads per slide
#   --batch-size 32   GPU batch size for Hibou-L forward pass
#   --dry-run         print plan without running
```

- Extracts tiles at **256 px / 128 µm** (≈ 20× magnification equivalent)
- Tissue segmentation via Otsu QC (auto-excludes background/whitespace)
- Feature extractor: **Hibou-L** (ViT-L/14, 1024-dim, frozen — no fine-tuning)
  - Weights download automatically from HuggingFace on first run
  - `slideflow-gpl` or the `histai/hibou-l` HF repo must be accessible
- Output: per-slide `.h5` bags in `./features/hibou_l/<slide_name>.h5`
  - Each bag: `features` (N×1024 float32) + `coords` (N×2 int32)
- **Estimated time:** ~2–4 min/slide on a single A100 → ~30h for 837 slides
  - Run on a GPU node; feature extraction is the bottleneck
- Incremental: already-extracted slides are skipped on re-run

### Step 2 — MIL training

```bash
python train_mil.py
# Key options:
#   --model     attention_mil | transmil | clam_sb | clam_mb  (default: attention_mil)
#   --folds     5             stratified k-fold CV (default: 5)
#   --epochs    20
#   --lr        1e-4
#   --pos-weight 5.07         BCE weight for class imbalance (LUAD 5.1:1 ratio)
#   --include-lusc            add LUSC slides to training set
```

- **Aggregation level:** patient (all slides per patient = one bag)
- **Splits:** patient-level stratified k-fold — no slide-level data leakage
- **Loss:** weighted binary cross-entropy (`pos_weight=5.07`)
- **Default model:** ABMIL (Ilse et al. 2018, attention-based)
- Models saved to `./mil/<experiment>/`

**Recommended order to try:**
1. `attention_mil` — ABMIL, standard baseline, fastest
2. `clam_sb` — CLAM single-branch, good for interpretability
3. `transmil` — transformer-based, better with large bag sizes

**Training notes:**
- With 62 driver patients and 5-fold CV, each fold trains on ~50 driver / ~284 WT patients
- This is a marginal dataset size — expect AUC 0.70–0.82 range
- If AUC < 0.70: check feature quality, try DX-only slides, consider augmentation
- If overfitting: reduce epochs, add dropout, try bag-size capping (e.g. `--bag-size 512`)

### Step 3 — Evaluation

```bash
python evaluate_mil.py --model-dir ./mil/00001-attention_mil
# Options:
#   --fold 0        evaluate specific fold (default: all folds)
#   --heatmaps      generate attention heatmaps (slow, needs GPU)
```

Outputs per experiment directory:
- `roc_curve.png` — per-fold ROC curves with AUC
- `prc_curve.png` — precision-recall curves with AUPRC
- `summary.json` — mean AUC, AUPRC, F1, sensitivity, specificity per fold

### Step 4 — Optional improvements

**If dataset is too small (AUC < 0.72):**
- Add LUSC driver cases (`--include-lusc`) — adds 4 more driver patients
- Use DX-only slides (`slide_type == DX`) — highest quality H&E
- External validation cohort: NLST or CPTAC-LUAD if accessible

**If class imbalance hurts recall:**
- Adjust `--pos-weight` upward (e.g. 8.0) to improve sensitivity at cost of specificity
- Undersample WT to 2:1 or 3:1 ratio — edit filter in `train_mil.py`

**Interpretability:**
- Run `evaluate_mil.py --heatmaps` to generate attention maps
- High-attention tiles in driver cases should localise to tumour nuclei / gland architecture
- Compare ABMIL vs CLAM attention distributions

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
