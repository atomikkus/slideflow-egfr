#!/usr/bin/env python3
"""
setup_project.py
----------------
Initializes the SlideFlow project for EGFR driver mutation classification.

Layout created:
  slideflow-egfr/
    project/                   ← SlideFlow project root
      slideflow.json           ← project config
      datasets/
        luad_lusc.json         ← GCS dataset config
    features/                  ← extracted feature bags (h5 files)
    mil/                       ← MIL model outputs

Usage:
    python setup_project.py [--project-dir ./project]
"""

import argparse
import json
from pathlib import Path

import slideflow as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path(__file__).parent / "project"

# GCS bucket sources (slideflow reads slides directly via gsutil/gcsfuse)
SLIDE_DIRS = [
    "gs://wsi_bucket53/TCGA_LUAD_SVS",
    "gs://wsi_bucket53/egfr_exon19_luad",
    "gs://wsi_bucket53/EGFR_SVS",
]

ANNOTATION_CSV = str(Path(__file__).parent / "annotations.csv")

# Tile extraction params
TILE_PX   = 256     # output tile size in pixels
TILE_UM   = 128     # physical size in microns (→ 20× magnification equivalent)
MPP       = 0.5     # microns per pixel (20× ≈ 0.5 mpp)

# Feature extractor
EXTRACTOR = "hibou_l"   # registered as 'hibou_l' in slideflow-gpl or via custom

# MIL aggregator options
MIL_MODEL = "attention_mil"   # ABMIL; alternatives: transmil, clam_sb, clam_mb


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------
def create_project(project_dir: Path) -> sf.Project:
    """Create (or open) a SlideFlow project."""
    project_dir.mkdir(parents=True, exist_ok=True)
    ann = ANNOTATION_CSV

    print(f"Creating SlideFlow project at {project_dir} …")
    P = sf.Project(
        root=str(project_dir),
        create=True,
    )
    # Configure annotations and slide directories after project creation
    P.annotations = ann
    for slide_dir in SLIDE_DIRS:
        P.add_source(name=slide_dir.split("/")[-1], slides=slide_dir,
                     tfrecords=str(project_dir / "tfrecords"))
    print(f"  Project root:  {P.root}")
    print(f"  Annotations:   {ann}")
    return P


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
def write_dataset_config(project_dir: Path):
    """Write a human-readable dataset config alongside the project."""
    config = {
        "name": "TCGA-LUAD-EGFR",
        "slides": SLIDE_DIRS,
        "annotations": ANNOTATION_CSV,
        "tile_px": TILE_PX,
        "tile_um": TILE_UM,
        "label_column": "egfr_driver",
        "patient_column": "patient",
        "slide_column": "slide",
        "filters": {
            "sample_type_code": ["01"],   # Primary Tumor only
            "egfr_driver": [0, 1],        # exclude Unknown (-1)
        },
        "class_names": {
            "0": "EGFR-Wildtype",
            "1": "EGFR-Driver",
        },
        "class_balance_note": (
            "Class imbalance ~1:4 (driver:WT). "
            "Use weighted loss (pos_weight≈3.85) or over-sampling."
        ),
    }
    out = project_dir / "datasets" / "luad_lusc.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Dataset config: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    P = create_project(project_dir)
    write_dataset_config(project_dir)

    print("\n=== Project ready ===")
    print("Next steps:")
    print("  1. Extract tiles + features:")
    print("     python extract_features.py")
    print("  2. Train MIL model:")
    print("     python train_mil.py")
    print("  3. Evaluate:")
    print("     python evaluate_mil.py")


if __name__ == "__main__":
    main()
