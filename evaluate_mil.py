#!/usr/bin/env python3
"""
evaluate_mil.py
---------------
Evaluate a trained MIL model and generate:
  - Patient-level ROC + PR curves
  - Per-fold AUC / AUPRC / F1 / sensitivity / specificity
  - Attention heatmaps (optional, requires slideflow-gpl or custom renderer)
  - Confusion matrix

Usage:
    python evaluate_mil.py --model-dir ./mil/00001-attention_mil \
                           [--fold 0] [--heatmaps]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        auc, average_precision_score, confusion_matrix,
        f1_score, precision_recall_curve, roc_curve,
    )
    SKL_OK = True
except ImportError:
    SKL_OK = False
    print("WARNING: scikit-learn not available — metrics will be limited")

import slideflow as sf
import slideflow.mil as sf_mil

# ---------------------------------------------------------------------------
REPO_DIR    = Path(__file__).parent
PROJECT_DIR = REPO_DIR / "project"
ANN_CSV     = REPO_DIR / "annotations.csv"
FEATURES_DIR = REPO_DIR / "features" / "hibou_l"
SLIDE_DIRS  = [
    "gs://wsi_bucket53/TCGA_LUAD_SVS",
    "gs://wsi_bucket53/egfr_exon19_luad",
    "gs://wsi_bucket53/EGFR_SVS",
]
LABEL_COL   = "egfr_driver"
PATIENT_COL = "patient"
TILE_PX     = 256
TILE_UM     = 128
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_score, threshold=0.5):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    y_pred = (np.array(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred)
    return {
        "AUC": roc_auc, "AUPRC": auprc,
        "F1": f1, "Sensitivity": sens, "Specificity": spec,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision": prec.tolist(), "recall": rec.tolist(),
    }


def plot_roc(metrics_list: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, m in enumerate(metrics_list):
        ax.plot(m["fpr"], m["tpr"],
                label=f"Fold {i+1} (AUC={m['AUC']:.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — EGFR Driver vs WT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC saved → {out_path}")


def plot_prc(metrics_list: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, m in enumerate(metrics_list):
        ax.plot(m["recall"], m["precision"],
                label=f"Fold {i+1} (AUPRC={m['AUPRC']:.3f})", lw=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — EGFR Driver vs WT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PRC saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Path to trained MIL experiment directory")
    ap.add_argument("--fold",      type=int, default=None,
                    help="Evaluate specific fold (0-indexed); default=all")
    ap.add_argument("--heatmaps",  action="store_true",
                    help="Generate attention heatmaps (slow)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.exists(), f"Model directory not found: {model_dir}"

    # Load project + dataset
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
        },
    )

    # Find fold result directories
    fold_dirs = sorted(model_dir.glob("fold*")) or [model_dir]
    if args.fold is not None:
        fold_dirs = [fold_dirs[args.fold]]

    all_metrics = []
    for fold_dir in fold_dirs:
        print(f"\nEvaluating {fold_dir.name} …")
        # Load predictions
        pred_csv = fold_dir / "predictions.csv"
        if not pred_csv.exists():
            print(f"  No predictions.csv found — skipping")
            continue
        pred_df = pd.read_csv(pred_csv)
        y_true  = pred_df["y_true"].values
        y_score = pred_df["y_pred"].values
        m = compute_metrics(y_true, y_score)
        all_metrics.append(m)
        print(f"  AUC={m['AUC']:.4f}  AUPRC={m['AUPRC']:.4f}  "
              f"F1={m['F1']:.4f}  Sens={m['Sensitivity']:.4f}  "
              f"Spec={m['Specificity']:.4f}")

    if all_metrics:
        # Plot
        plot_roc(all_metrics, model_dir / "roc_curve.png")
        plot_prc(all_metrics, model_dir / "prc_curve.png")

        # Summary JSON
        summary = {
            "mean_AUC":   np.mean([m["AUC"]   for m in all_metrics]),
            "mean_AUPRC": np.mean([m["AUPRC"] for m in all_metrics]),
            "mean_F1":    np.mean([m["F1"]    for m in all_metrics]),
            "folds": [{k: v for k, v in m.items()
                       if k not in ("fpr","tpr","precision","recall")}
                      for m in all_metrics],
        }
        with open(model_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Mean AUC:   {summary['mean_AUC']:.4f}")
        print(f"  Mean AUPRC: {summary['mean_AUPRC']:.4f}")
        print(f"  Summary → {model_dir / 'summary.json'}")

    if args.heatmaps:
        print("\nGenerating attention heatmaps …")
        for fold_dir in fold_dirs:
            P.generate_heatmaps(
                model=str(fold_dir / "model.pth"),
                dataset=dataset,
                outdir=str(fold_dir / "heatmaps"),
                batch_size=32,
            )


if __name__ == "__main__":
    main()
