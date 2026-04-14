#!/usr/bin/env python3
"""
evaluate_mil.py
---------------
Evaluate a trained MIL model by aggregating out-of-fold predictions
and plotting ROC + precision-recall curves.

Reads:  <outdir>/oof_predictions.parquet   (pooled across all folds)
        <outdir>/fold*/predictions.parquet  (per-fold, for individual curves)

Outputs:
    <outdir>/roc_curve.png   — aggregated OOF ROC + per-fold ROC curves
    <outdir>/prc_curve.png   — aggregated OOF PRC + per-fold PRC curves
    <outdir>/summary.json    — AUC, AUPRC, F1, sensitivity, specificity

Usage:
    python evaluate_mil.py [--outdir ./mil] [--model-tag attention_mil-egfr_driver]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def best_threshold_f1(y_true, y_pred):
    """Return threshold that maximises F1 on the given predictions."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = 2 * prec * rec / np.where((prec + rec) == 0, 1, prec + rec)
    return float(thresholds[np.argmax(f1[:-1])])


def find_fold_preds(out_dir: Path, model_tag: str):
    """Locate predictions.parquet files for each fold, sorted by fold number."""
    results = {}
    for fold_dir in sorted(out_dir.glob("fold*")):
        matches = list(fold_dir.rglob(f"*{model_tag}*/predictions.parquet"))
        if not matches:
            matches = list(fold_dir.rglob("predictions.parquet"))
        if matches:
            fold_num = int(fold_dir.name.replace("fold", ""))
            results[fold_num] = matches[0]
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",    default="./mil",
                    help="MIL output directory (contains fold1/, fold2/, …)")
    ap.add_argument("--model-tag", default="attention_mil-egfr_driver",
                    help="Substring to match the model subdirectory name")
    args = ap.parse_args()

    out_dir = Path(args.outdir)

    # ---- Load OOF predictions ----
    oof_path = out_dir / "oof_predictions.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path)
        print(f"Loaded OOF predictions: {len(oof_df)} patients "
              f"({int(oof_df['y_true'].sum())} driver, "
              f"{int((oof_df['y_true'] == 0).sum())} WT)")
    else:
        # Fallback: assemble from per-fold parquets
        fold_preds = find_fold_preds(out_dir, args.model_tag)
        if not fold_preds:
            print(f"No predictions found in {out_dir}. Run train_mil.py first.")
            return
        dfs = []
        for fold_num, path in fold_preds.items():
            df = pd.read_parquet(path)
            df["fold"] = fold_num
            dfs.append(df)
        oof_df = pd.concat(dfs, ignore_index=True)
        oof_df.to_parquet(oof_path, index=False)
        print(f"Assembled OOF from {len(fold_preds)} fold(s): {len(oof_df)} patients")

    # ---- Per-fold parquets for individual curves ----
    fold_preds = find_fold_preds(out_dir, args.model_tag)

    # ---- Aggregate OOF metrics ----
    y_true = oof_df["y_true"].values
    y_pred = oof_df["y_pred1"].values

    oof_auc  = roc_auc_score(y_true, y_pred)
    oof_ap   = average_precision_score(y_true, y_pred)
    thresh   = best_threshold_f1(y_true, y_pred)
    y_binary = (y_pred >= thresh).astype(int)
    oof_f1   = f1_score(y_true, y_binary)
    oof_sens = float(y_binary[y_true == 1].mean())
    oof_spec = float((1 - y_binary[y_true == 0]).mean())

    print(f"\n=== Aggregated OOF Results ===")
    print(f"  AUC        : {oof_auc:.3f}")
    print(f"  AUPRC      : {oof_ap:.3f}")
    print(f"  F1 (opt θ) : {oof_f1:.3f}  (threshold={thresh:.3f})")
    print(f"  Sensitivity: {oof_sens:.3f}")
    print(f"  Specificity: {oof_spec:.3f}")

    # Per-fold metrics
    fold_metrics = []
    for fold_num, path in sorted(fold_preds.items()):
        fd = pd.read_parquet(path)
        fa = roc_auc_score(fd["y_true"], fd["y_pred1"])
        fp = average_precision_score(fd["y_true"], fd["y_pred1"])
        fold_metrics.append({"fold": fold_num, "auc": round(fa, 4), "auprc": round(fp, 4),
                              "n_patients": len(fd), "n_driver": int(fd["y_true"].sum())})
        print(f"  Fold {fold_num}: AUC={fa:.3f}  AUPRC={fp:.3f}  "
              f"({int(fd['y_true'].sum())} driver / {len(fd)} patients)")

    # ---- ROC curve ----
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr_oof, tpr_oof, _ = roc_curve(y_true, y_pred)
    ax.plot(fpr_oof, tpr_oof, color="black", lw=2.5,
            label=f"OOF aggregate (AUC={oof_auc:.3f})")

    colors = plt.cm.tab10.colors
    for fm in fold_metrics:
        fd = pd.read_parquet(fold_preds[fm["fold"]])
        fpr, tpr, _ = roc_curve(fd["y_true"], fd["y_pred1"])
        ax.plot(fpr, tpr, lw=1, alpha=0.6, color=colors[fm["fold"] - 1],
                label=f"Fold {fm['fold']} (AUC={fm['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("EGFR Driver Mutation — ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"\n  ROC curve → {roc_path}")

    # ---- PRC curve ----
    fig, ax = plt.subplots(figsize=(6, 6))
    prec_oof, rec_oof, _ = precision_recall_curve(y_true, y_pred)
    baseline = y_true.mean()
    ax.plot(rec_oof, prec_oof, color="black", lw=2.5,
            label=f"OOF aggregate (AUPRC={oof_ap:.3f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=0.8,
               label=f"Random baseline ({baseline:.3f})")

    for fm in fold_metrics:
        fd = pd.read_parquet(fold_preds[fm["fold"]])
        prec, rec, _ = precision_recall_curve(fd["y_true"], fd["y_pred1"])
        ax.plot(rec, prec, lw=1, alpha=0.6, color=colors[fm["fold"] - 1],
                label=f"Fold {fm['fold']} (AUPRC={fm['auprc']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("EGFR Driver Mutation — Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    prc_path = out_dir / "prc_curve.png"
    fig.savefig(prc_path, dpi=150)
    plt.close(fig)
    print(f"  PRC curve  → {prc_path}")

    # ---- Summary JSON ----
    summary = {
        "model": args.model_tag,
        "n_patients": int(len(oof_df)),
        "n_driver": int(y_true.sum()),
        "n_wt": int((y_true == 0).sum()),
        "oof_auc": round(oof_auc, 4),
        "oof_auprc": round(oof_ap, 4),
        "oof_f1": round(oof_f1, 4),
        "oof_sensitivity": round(oof_sens, 4),
        "oof_specificity": round(oof_spec, 4),
        "opt_threshold": round(thresh, 4),
        "folds": fold_metrics,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary    → {summary_path}")


if __name__ == "__main__":
    main()
