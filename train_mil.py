#!/usr/bin/env python3
"""
train_mil.py
------------
Train a Multiple Instance Learning (MIL) model on DINOv2 ViT-L/14 features
for EGFR driver mutation classification in TCGA-LUAD.

Model: Attention-based MIL (ABMIL) — Ilse et al. 2018
       Optionally: TransMIL, CLAM-SB, CLAM-MB

Label:   egfr_driver   (1 = FDA/NCCN driver mutation, 0 = WT / non-driver)
Inputs:  pre-extracted DINOv2 feature bags (*.pt)

Training strategy:
  - Patient-level stratified 3-fold cross-validation
  - Out-of-fold predictions pooled → single AUC over full cohort
  - Weighted cross-entropy loss (weighted_loss=True, handles 5:1 imbalance)
  - AdamW optimizer, fastai one-cycle LR schedule
  - Frozen feature extractor (DINOv2, no fine-tuning)
  - Patient-level train/val splits (no slide-level leakage)

Usage:
    python train_mil.py [--model attention_mil] [--folds 3] [--epochs 20]
                        [--lr 1e-4] [--wd 1e-5] [--outdir ./mil]
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
FEATURES_DIR = REPO_DIR / "features" / "dinov2_vitl14"
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
    ap.add_argument("--folds",      type=int, default=3,
                    help="Stratified k-fold CV (default: 3 — gives ~21 driver "
                         "patients per val fold vs ~12 with 5-fold)")
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--wd",         type=float, default=1e-5)
    ap.add_argument("--outdir",     default=str(MIL_DIR))
    ap.add_argument("--include-lusc", action="store_true",
                    help="Include LUSC slides (default: LUAD-only)")
    ap.add_argument("--dx-only",    action="store_true",
                    help="Use DX (diagnostic) slides only — highest quality H&E")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load annotations ----
    df = pd.read_csv(ANN_CSV, dtype={"sample_type_code": str})
    df = df[
        (df["sample_type_code"] == "01") &
        (df[LABEL_COL].isin([0, 1]))
    ]
    if not args.include_lusc:
        df = df[df["cancer_type"] == "LUAD"]
        print(f"LUAD-only mode: {len(df)} slides (use --include-lusc to add LUSC)")
    if args.dx_only:
        df = df[df["slide_type"] == "DX"]
        print(f"DX-only mode: {len(df)} slides")

    print(f"\nTraining set: {len(df)} slides, "
          f"{df[LABEL_COL].sum()} driver, "
          f"{(df[LABEL_COL] == 0).sum()} WT, "
          f"{df[PATIENT_COL].nunique()} patients")

    # ---- Open project ----
    if (PROJECT_DIR / "settings.json").exists():
        P = sf.load_project(str(PROJECT_DIR))
    else:
        P = sf.Project(
            root=str(PROJECT_DIR),
            annotations=str(ANN_CSV),
            slides=SLIDE_DIRS,
        )

    dataset_filters = {
        "sample_type_code": ["01"],
        LABEL_COL: ["0", "1"],
        **({"cancer_type": ["LUAD"]} if not args.include_lusc else {}),
        **({"slide_type": ["DX"]}    if args.dx_only else {}),
    }
    dataset = P.dataset(
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        filters=dataset_filters,
    )

    # ---- MIL config ----
    mil_config = sf_mil.mil_config(
        model=args.model,
        aggregation_level="patient",
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        bag_size=512,
        weighted_loss=True,
    )

    # ---- Patient-level stratified k-fold CV ----
    from sklearn.model_selection import StratifiedKFold
    patients = df.groupby(PATIENT_COL)[LABEL_COL].max().reset_index()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    oof_preds = []   # out-of-fold predictions — pooled across all folds
    print(f"\nStarting {args.folds}-fold CV with {args.model} …")
    print(f"Driver patients: {patients[LABEL_COL].sum()} | "
          f"WT patients: {(patients[LABEL_COL] == 0).sum()}")
    print(f"Expected ~{patients[LABEL_COL].sum() // args.folds} driver patients per val fold\n")

    for fold_i, (train_idx, val_idx) in enumerate(
        skf.split(patients[PATIENT_COL], patients[LABEL_COL])
    ):
        train_patients = patients.iloc[train_idx][PATIENT_COL].tolist()
        val_patients   = patients.iloc[val_idx][PATIENT_COL].tolist()

        n_train_driver = int(patients.iloc[train_idx][LABEL_COL].sum())
        n_val_driver   = int(patients.iloc[val_idx][LABEL_COL].sum())

        print(f"--- Fold {fold_i + 1}/{args.folds} ---")
        print(f"  Train: {len(train_patients)} patients ({n_train_driver} driver) | "
              f"Val: {len(val_patients)} patients ({n_val_driver} driver)")

        train_ds = dataset.filter(filters={PATIENT_COL: train_patients})
        val_ds   = dataset.filter(filters={PATIENT_COL: val_patients})

        fold_dir = out_dir / f"fold{fold_i + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        learner = sf_mil.train_mil(
            config=mil_config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            outcomes=LABEL_COL,
            bags=str(FEATURES_DIR),
            outdir=str(fold_dir),
        )

        # Collect out-of-fold predictions
        pred_files = list(fold_dir.rglob("predictions.parquet"))
        if pred_files:
            fold_preds = pd.read_parquet(pred_files[0])
            fold_preds["fold"] = fold_i + 1
            oof_preds.append(fold_preds)
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc  = roc_auc_score(fold_preds["y_true"], fold_preds["y_pred1"])
            ap   = average_precision_score(fold_preds["y_true"], fold_preds["y_pred1"])
            print(f"  Fold {fold_i + 1} AUC: {auc:.3f}  AUPRC: {ap:.3f}")

    # ---- Pool out-of-fold predictions ----
    if oof_preds:
        oof_df = pd.concat(oof_preds, ignore_index=True)
        oof_path = out_dir / "oof_predictions.parquet"
        oof_df.to_parquet(oof_path, index=False)

        from sklearn.metrics import roc_auc_score, average_precision_score
        oof_auc = roc_auc_score(oof_df["y_true"], oof_df["y_pred1"])
        oof_ap  = average_precision_score(oof_df["y_true"], oof_df["y_pred1"])

        print(f"\n=== Cross-Validation Complete ===")
        print(f"  Folds: {args.folds}  |  Patients evaluated: {len(oof_df)}")
        print(f"  Aggregated OOF AUC  : {oof_auc:.3f}")
        print(f"  Aggregated OOF AUPRC: {oof_ap:.3f}")
        print(f"  OOF predictions → {oof_path}")
        print(f"  Saved models     → {out_dir}")


if __name__ == "__main__":
    main()
