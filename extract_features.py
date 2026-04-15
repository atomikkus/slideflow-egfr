#!/usr/bin/env python3
"""
extract_features.py
-------------------
Tile and feature-extract all primary-tumor slides using DINOv2 ViT-L/14.

Features are saved as per-slide bags in:
    ./features/<extractor_name>/<slide_name>.pt

Each .pt contains:
    (N, D) float32 feature vectors  (D=1024 for DINOv2)

Usage:
    python extract_features.py [--workers 4] [--batch-size 32] [--dry-run]
                               [--batch-slides 150] [--normalizer macenko]

Notes:
    - Slides are read from gcsfuse-mounted GCS bucket (slides/wsi_bucket53/)
    - Tiles are extracted at 256 px / 128 µm (≈ 20× equivalent)
    - DINOv2 weights are downloaded automatically via torch.hub
    - Already-extracted slides are skipped (incremental)
    - TFRecords are deleted after features are confirmed saved (saves ~384 MB/slide)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import slideflow as sf
from slideflow.model import build_feature_extractor
from dinov2_extractor import register_dinov2_vitl

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_DIR     = Path(__file__).parent
PROJECT_DIR  = REPO_DIR / "project"
ANN_CSV      = REPO_DIR / "annotations.csv"
FEATURES_DIR = REPO_DIR / "features"
TFRECORDS_DIR = PROJECT_DIR / "tfrecords" / "256px_128um"

TILE_PX  = 256
TILE_UM  = 128
QC       = "otsu"
EXTRACTOR = "dinov2_vitl14"

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
    df = df[(df["sample_type_code"] == 1) & (df["egfr_driver"].isin([0, 1]))]
    print(f"Training slides: {len(df)} ({df['egfr_driver'].sum()} driver, "
          f"{(df['egfr_driver'] == 0).sum()} WT)")
    return df


def already_extracted(slide_name: str, features_dir: Path, extractor: str) -> bool:
    pt = features_dir / extractor / f"{slide_name}.pt"
    return pt.exists() and pt.stat().st_size > 0


def cleanup_tfrecords(slide_names: list, dry_run: bool = False) -> int:
    """Delete TFRecords for slides that have confirmed .pt feature bags.

    Only deletes when the .pt bag exists and is non-empty — never deletes
    a TFRecord if extraction may not have completed.
    """
    deleted = 0
    for slide_name in slide_names:
        tfr = TFRECORDS_DIR / f"{slide_name}.tfrecords"
        pt = FEATURES_DIR / EXTRACTOR / f"{slide_name}.pt"
        if tfr.exists() and pt.exists() and pt.stat().st_size > 0:
            if not dry_run:
                tfr.unlink()
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers",       type=int, default=4,
                    help="Tile-extraction worker threads per slide")
    ap.add_argument("--batch-size",    type=int, default=32,
                    help="GPU batch size for DINOv2 forward pass")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Print plan without extracting or deleting")
    ap.add_argument("--sample",        type=int, default=None,
                    help="Randomly sample N slides per class (balanced subset)")
    ap.add_argument("--seed",          type=int, default=42,
                    help="Random seed for --sample (default: 42)")
    ap.add_argument("--batch-slides",  type=int, default=150,
                    help="Process this many slides per batch, deleting TFRecords "
                         "after each batch to save disk space (default: 150)")
    ap.add_argument("--normalizer",    type=str, default="macenko",
                    choices=["macenko", "reinhard", "none"],
                    help="Stain normalizer for tile extraction "
                         "(default: macenko, use 'none' to disable)")
    args = ap.parse_args()
    normalizer = None if args.normalizer == "none" else args.normalizer

    register_dinov2_vitl()
    df = load_training_slides(ANN_CSV)

    if args.sample is not None:
        pos = df[df["egfr_driver"] == 1]
        neg = df[df["egfr_driver"] == 0]
        n = args.sample
        if n > len(pos):
            print(f"Warning: only {len(pos)} driver slides available, sampling all.")
            n = len(pos)
        pos_sample = pos.sample(n=n, random_state=args.seed)
        neg_sample = neg.sample(n=n, random_state=args.seed)
        df = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
        print(f"Sampled subset: {len(pos_sample)} driver + {len(neg_sample)} WT = {len(df)} slides")

    # Write temp annotations CSV if sampling
    tmp_ann = REPO_DIR / "_tmp_annotations.csv"
    if args.sample is not None:
        df_out = df.copy()
        df_out["sample_type_code"] = df_out["sample_type_code"].apply(lambda x: f"{int(x):02d}")
        df_out.to_csv(tmp_ann, index=False)

    ann_to_use = str(tmp_ann) if args.sample is not None else str(ANN_CSV)
    if (PROJECT_DIR / "settings.json").exists():
        P = sf.load_project(str(PROJECT_DIR))
        P._settings['annotations'] = ann_to_use
    else:
        P = sf.Project(
            root=str(PROJECT_DIR),
            annotations=ann_to_use,
            slides=SLIDE_DIRS,
        )

    tile_filters = {
        "sample_type_code": ["01"],
        "egfr_driver": ["0", "1"],
    }

    dataset = P.dataset(
        tile_px=TILE_PX,
        tile_um=TILE_UM,
        filters=tile_filters,
    )

    all_slides = dataset.slides()
    print(f"\nDataset: {len(all_slides)} slides total")
    print(f"Tile stain normalizer: {normalizer or 'disabled'}")

    # ---- Upfront cleanup: delete TFRecords for already-extracted slides ----
    already_done = [s for s in all_slides if already_extracted(s, FEATURES_DIR, EXTRACTOR)]
    print(f"Already extracted: {len(already_done)} slides")
    if already_done:
        n_cleaned = cleanup_tfrecords(already_done, dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"{action} {n_cleaned} stale TFRecords for already-extracted slides")

    todo_slides = [s for s in all_slides if not already_extracted(s, FEATURES_DIR, EXTRACTOR)]
    print(f"Remaining to extract: {len(todo_slides)} slides")

    if args.dry_run:
        print("Dry-run mode: exiting without extraction.")
        return

    if not todo_slides:
        print("All slides already extracted. Nothing to do.")
        return

    # ---- Build extractor once ----
    import sys
    print(f"\nBuilding feature extractor: {EXTRACTOR} …", flush=True)
    extractor = build_feature_extractor(EXTRACTOR, tile_px=TILE_PX)
    print(f"Feature extractor ready. num_features={extractor.num_features}", flush=True)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Process in batches ----
    batch_sz = args.batch_slides
    n_batches = (len(todo_slides) + batch_sz - 1) // batch_sz
    print(f"\nProcessing {len(todo_slides)} slides in {n_batches} batch(es) of ≤{batch_sz}", flush=True)

    for batch_i in range(n_batches):
        batch = todo_slides[batch_i * batch_sz : (batch_i + 1) * batch_sz]
        print(f"\n=== Batch {batch_i + 1}/{n_batches}: {len(batch)} slides ===", flush=True)

        batch_filters = {**tile_filters, "slide": batch}
        batch_ds = dataset.filter(filters={"slide": batch})

        # Step 1 — Tile extraction
        existing_tfrs = set(
            Path(f).stem for f in TFRECORDS_DIR.glob("*.tfrecords")
        ) if TFRECORDS_DIR.exists() else set()
        needs_tiling = [s for s in batch if s not in existing_tfrs]
        print(f"  Slides needing tiling: {len(needs_tiling)}/{len(batch)}", flush=True)

        if needs_tiling:
            print("  Extracting tiles → TFRecords …", flush=True)
            P.extract_tiles(
                tile_px=TILE_PX,
                tile_um=TILE_UM,
                filters=batch_filters,
                qc=QC,
                normalizer=normalizer,
                num_threads=args.workers,
                report=False,
            )

        # Step 2 — Feature extraction
        print(f"  Generating feature bags → {FEATURES_DIR / EXTRACTOR} …", flush=True)
        sys.stdout.flush()
        P.generate_feature_bags(
            extractor,
            dataset=batch_ds,
            outdir=str(FEATURES_DIR / EXTRACTOR),
        )

        # Step 3 — Safe TFRecord cleanup (only if .pt exists and is non-empty)
        n_deleted = cleanup_tfrecords(batch, dry_run=False)
        print(f"  Cleaned up {n_deleted}/{len(batch)} TFRecords (confirmed bags exist)", flush=True)

        bags_done = len(list((FEATURES_DIR / EXTRACTOR).glob("*.pt")))
        print(f"  Total bags so far: {bags_done}", flush=True)

    print("\nFeature extraction complete.", flush=True)
    bags = list((FEATURES_DIR / EXTRACTOR).glob("*.pt"))
    print(f"  Total bags: {len(bags)}", flush=True)


if __name__ == "__main__":
    main()
