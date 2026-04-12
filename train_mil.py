#!/usr/bin/env python3
"""
train_mil.py
------------
Train a Multiple Instance Learning (MIL) model on Hibou-L features
for EGFR driver mutation classification in TCGA-LUAD/LUSC.

Model: Attention-based MIL (ABMIL) — James Ilse et al. 2018
       Optionally: TransMIL, CLAM-SB, CLAM-MB

Label:   egfr_driver   (1 = FDA/NCCN driver mutation, 0 = WT / non-driver)
Inputs:  pre-extracted Hibou-L feature bags (*.h5)

Training strategy:
  - Stratified 5-fold cross-validation
  - Weighted BCE loss (pos_weight ≈ 3.85 for ~1:4 imbalance)
  - AdamW optimizer, cosine LR schedule
  - Frozen feature extractor (Hibou-L, no fine-tuning)
  - Patient-level train/val/test splits (no slide-level leakage)

Usage:
    python train_mil.py [--model attention_mil] [--folds 5] [--epochs 20]
                        [--lr 1e-4] [--wd 1e-5] [--pos-weight 3.85]
                        [--outdir ./mil]
"""

import argparse
from pathlib import Path

import pandas as pd
import slideflow as sf
import slideflow.mil as sf_mil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_DIR     = Path(__file__).parent
PROJECT_DIR  = REPO_DIR / "project"
ANN_CSV      = REPO_DIR / "annotations.csv"
FEATURES_DIR = REPO_DIR / "features" / "hibou_l"
MIL_DIR      = REPO_DIR / "mil"

SLIDE_DIRS = [
    "gs://wsi_bucket53/TCGA_LUAD_SVS",
    "gs://wsi_bucket53/egfr_exon19_luad",
    "gs://wsi_bucket53/EGFR_SVS",
]

LABEL_COL   = "egfr_driver"
PATIENT_COL = "patient"
TILE_PX     = 256
TILE_UM     = 128


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="attention_mil",
                    choices=["attention_mil", "transmil", "clam_sb", "clam_mb"],
                    help="MIL aggregator architecture")
    ap.add_argument("--folds",      type=int, default=5)
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--wd",         type=float, default=1e-5)
    ap.add_argument("--pos-weight", type=float, default=3.85,
                    help="BCE positive class weight (WT:Driver ratio ≈ 3.85)")
    ap.add_argument("--outdir",     default=str(MIL_DIR))
    ap.add_argument("--luad-only",  action="store_true",
                    help="Restrict to LUAD slides (exclude LUSC)")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load annotations ----
    df = pd.read_csv(ANN_CSV)
    df = df[
        (df["sample_type_code"] == "01") &
        (df[LABEL_COL].isin([0, 1]))
    ]
    if args.luad_only:
        df = df[df["cancer_type"] == "LUAD"]
        print(f"LUAD-only mode: {len(df)} slides")

    print(f"\nTraining set: {len(df)} slides, "
          f"{df[LABEL_COL].sum()} driver, "
          f"{(df[LABEL_COL] == 0).sum()} WT, "
          f"{df[PATIENT_COL].nunique()} patients")

    # ---- Open project ----
    P = sf.Project(
        root=str(PROJECT_DIR),
        annotations=str(ANN_CSV),
        slides=SLIDE_DIRS,
    )

    dataset = P.dataset(
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        filters={
            "sample_type_code": ["01"],
            LABEL_COL: [0, 1],
            **({"cancer_type": ["LUAD"]} if args.luad_only else {}),
        },
    )

    # ---- MIL config ----
    mil_config = sf_mil.mil_config(
        model=args.model,
        aggregation_level="patient",   # bag = all slides per patient
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        bag_size=None,                 # use full bag (no truncation)
        # Weighted BCE for class imbalance
        loss="binary_crossentropy",
        weighted_loss=True,
        pos_weight=args.pos_weight,
    )

    # ---- Cross-validation ----
    print(f"\nStarting {args.folds}-fold CV with {args.model} …")
    results = P.train_mil(
        config=mil_config,
        outcomes=LABEL_COL,
        train_dataset=dataset,
        bags=str(FEATURES_DIR),
        outdir=str(out_dir),
        val_strategy="k-fold",
        val_k_fold=args.folds,
        # Patient-level splits (critical — prevents data leakage)
        splits=PATIENT_COL,
    )

    # ---- Print summary ----
    print("\n=== Cross-Validation Results ===")
    for fold_i, r in enumerate(results):
        print(f"  Fold {fold_i + 1}: AUC = {r.get('auc', float('nan')):.4f}  "
              f"AUPRC = {r.get('auprc', float('nan')):.4f}")

    aucs = [r.get("auc", 0) for r in results]
    print(f"\n  Mean AUC:   {sum(aucs)/len(aucs):.4f}")
    print(f"  Saved models → {out_dir}")


if __name__ == "__main__":
    main()
