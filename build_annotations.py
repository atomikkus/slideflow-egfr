#!/usr/bin/env python3
"""
build_annotations.py
--------------------
Builds the SlideFlow training annotation CSV from:
  - GCS slide inventory across three bucket folders
  - cBioPortal mutation data stored in the tcga-luad-egfr Excel outputs

Output: annotations.csv (one row per TCGA slide)
Columns:
  patient         – TCGA 12-char patient barcode (TCGA-XX-XXXX)
  slide           – slide filename without extension (slideflow key)
  gcs_path        – full gs:// URI to the SVS file
  source_bucket   – bucket folder tag (TCGA_LUAD_SVS / egfr_exon19_luad / EGFR_SVS)
  sample_id       – 16-char TCGA sample barcode (TCGA-XX-XXXX-XXX)
  sample_type_code– 01/11/06 etc (01=Primary Tumor, 11=Solid Normal)
  sample_type     – "Primary Tumor" / "Normal" / …
  slide_type      – DX / TS / BS
  cancer_type     – LUAD / LUSC (from GDC cBioPortal)
  egfr_driver     – 1 if FDA/NCCN driver mutation, else 0  ← primary label
  egfr_mutated    – 1 if any EGFR mutation (driver + VUS/other), else 0
  protein_change  – comma-separated list of all EGFR protein changes for patient
  driver_category – comma-separated driver categories (blank if WT)
  egfr_class      – blank/Driver/VUS-Missense/Nonsense/Frameshift/Splice
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).parent
TCGA_LUAD_EGFR_DIR = Path("/home/satya/Github/tcga-luad-egfr")
DISCOVERY_XLSX = TCGA_LUAD_EGFR_DIR / "egfr_gdc_discovery.xlsx"
MUTATION_XLSX = TCGA_LUAD_EGFR_DIR / "egfr_mutation_status.xlsx"
OUT_CSV = REPO_DIR / "annotations.csv"

GCS_BUCKETS = {
    "TCGA_LUAD_SVS":    "gs://wsi_bucket53/TCGA_LUAD_SVS/",
    "egfr_exon19_luad": "gs://wsi_bucket53/egfr_exon19_luad/",
    "EGFR_SVS":         "gs://wsi_bucket53/EGFR_SVS/",
}

# TCGA barcode pattern embedded in filenames
TCGA_RE = re.compile(
    r"(TCGA-\w{2}-\w{4})"       # patient id (12 chars)
    r"-(\d{2}[A-Z])"            # sample type code e.g. 01A
    r"-\d+[A-Z]?"               # vial + portion
    r"-([A-Z]{2}\d+)"           # slide code e.g. DX1, TS1, BS1
    r"\.",                      # dot before UUID or extension
    re.IGNORECASE,
)

SAMPLE_TYPE_MAP = {
    "01": "Primary Tumor",
    "02": "Recurrent Tumor",
    "06": "Metastatic",
    "11": "Solid Tissue Normal",
    "12": "Buccal Cell Normal",
    "14": "Bone Marrow Normal",
}


# ---------------------------------------------------------------------------
# Step 1 – enumerate all SVS files from GCS
# ---------------------------------------------------------------------------
def list_gcs_svs(prefix: str) -> list[str]:
    print(f"  Listing {prefix} …", end=" ", flush=True)
    r = subprocess.run(
        ["gsutil", "ls", "-r", f"{prefix}**.svs"],
        capture_output=True, text=True, timeout=120,
    )
    paths = [l.strip() for l in r.stdout.splitlines() if l.strip().endswith(".svs")]
    print(f"{len(paths)} files")
    return paths


def parse_tcga_filename(gcs_path: str) -> dict | None:
    """Extract structured fields from a TCGA SVS filename."""
    filename = gcs_path.split("/")[-1]
    m = TCGA_RE.search(filename)
    if not m:
        return None
    patient_id = m.group(1).upper()
    sample_code_full = m.group(2).upper()   # e.g. '01A'
    sample_type_code = sample_code_full[:2].zfill(2)  # numeric part e.g. '01'
    slide_type = m.group(3)[:2].upper()     # DX / TS / BS

    # Reconstruct full 16-char sample barcode from filename
    # TCGA-XX-XXXX-01A-01-DX1... → sample_id = TCGA-XX-XXXX-01A
    idx = gcs_path.find(patient_id)
    chunk = gcs_path[idx: idx + 16]
    sample_id = chunk if len(chunk) == 16 else patient_id + "-" + sample_code_full

    return {
        "patient":           patient_id,
        "sample_id":         sample_id,
        "sample_type_code":  sample_type_code,
        "sample_type":       SAMPLE_TYPE_MAP.get(sample_type_code, f"Code-{sample_type_code}"),
        "slide_type":        slide_type,
        "slide":             filename.rsplit(".", 1)[0],   # no extension
        "gcs_path":          gcs_path,
    }


def build_slide_inventory() -> pd.DataFrame:
    print("\n[1/3] Building slide inventory from GCS …")
    rows = []
    for bucket_tag, prefix in GCS_BUCKETS.items():
        paths = list_gcs_svs(prefix)
        for p in paths:
            parsed = parse_tcga_filename(p)
            if parsed is None:
                continue  # skip non-TCGA clinical files in EGFR_SVS
            parsed["source_bucket"] = bucket_tag
            rows.append(parsed)
    df = pd.DataFrame(rows)
    print(f"  → {len(df)} TCGA slides parsed "
          f"({df['patient'].nunique()} unique patients)")
    return df


# ---------------------------------------------------------------------------
# Step 2 – load mutation labels
# ---------------------------------------------------------------------------
def load_mutation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        driver_df  – rows from Driver_All_GDC (patient_id, cancer_type,
                     protein_change, driver_category)
        other_df   – rows from Other_EGFR_All_GDC (patient_id, cancer_type,
                     protein_change, egfr_class)
    """
    print("\n[2/3] Loading mutation data …")
    xl = pd.ExcelFile(DISCOVERY_XLSX)

    driver_df = xl.parse("Driver_All_GDC")[
        ["patient_id", "cancer_type", "protein_change", "driver_category"]
    ].copy()
    other_df = xl.parse("Other_EGFR_All_GDC")[
        ["patient_id", "cancer_type", "protein_change", "egfr_class"]
    ].copy()

    # Also check egfr_mutation_status.xlsx for LUAD WT cancer_type assignment
    xl2 = pd.ExcelFile(MUTATION_XLSX)
    wt_df = xl2.parse("EGFR_Wild_Type")[["patient_id"]].copy()
    wt_df["cancer_type"] = "LUAD"   # all WT patients from TCGA_LUAD_SVS bucket

    print(f"  Driver patients: {driver_df['patient_id'].nunique()}")
    print(f"  Other-EGFR patients: {other_df['patient_id'].nunique()}")
    print(f"  WT patients: {wt_df['patient_id'].nunique()}")
    return driver_df, other_df, wt_df


# ---------------------------------------------------------------------------
# Step 3 – merge slides with mutation labels
# ---------------------------------------------------------------------------
def merge_labels(
    slides: pd.DataFrame,
    driver_df: pd.DataFrame,
    other_df: pd.DataFrame,
    wt_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[3/3] Merging slides with mutation labels …")

    # ---- Aggregate per-patient mutation info ----
    # Driver: aggregate protein_change and driver_category
    driver_agg = (
        driver_df.groupby("patient_id")
        .agg(
            cancer_type=("cancer_type", "first"),
            protein_change=("protein_change", lambda x: ", ".join(sorted(set(x)))),
            driver_category=("driver_category", lambda x: " | ".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"patient_id": "patient"})
    )
    driver_agg["egfr_driver"]  = 1
    driver_agg["egfr_mutated"] = 1
    driver_agg["egfr_class"]   = "Driver"

    # Other (VUS/non-driver): aggregate
    other_agg = (
        other_df.groupby("patient_id")
        .agg(
            cancer_type=("cancer_type", "first"),
            protein_change=("protein_change", lambda x: ", ".join(sorted(set(x)))),
            egfr_class=("egfr_class", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"patient_id": "patient"})
    )
    other_agg["egfr_driver"]   = 0
    other_agg["egfr_mutated"]  = 1
    other_agg["driver_category"] = ""

    # WT
    wt_agg = wt_df.drop_duplicates("patient_id").rename(columns={"patient_id": "patient"})
    wt_agg["egfr_driver"]    = 0
    wt_agg["egfr_mutated"]   = 0
    wt_agg["protein_change"] = ""
    wt_agg["driver_category"] = ""
    wt_agg["egfr_class"]     = "WT"

    # Combine – driver takes priority if a patient appears in both
    driver_patients = set(driver_agg["patient"])
    other_agg_filtered = other_agg[~other_agg["patient"].isin(driver_patients)]
    mut_patients = driver_patients | set(other_agg["patient"])
    wt_agg_filtered = wt_agg[~wt_agg["patient"].isin(mut_patients)]

    mut_df = pd.concat(
        [driver_agg, other_agg_filtered, wt_agg_filtered],
        ignore_index=True,
        sort=False,
    )

    # ---- Merge slides with mutation info ----
    merged = slides.merge(mut_df, on="patient", how="left")

    # Fill unknowns (patients in bucket but not in cBioPortal queries)
    unknown_mask = merged["egfr_driver"].isna()
    n_unknown = unknown_mask.sum()
    if n_unknown:
        print(f"  WARNING: {n_unknown} slides have no cBioPortal match "
              f"(treated as Unknown/excluded from training set)")
        merged.loc[unknown_mask, "egfr_driver"]   = -1
        merged.loc[unknown_mask, "egfr_mutated"]  = -1
        merged.loc[unknown_mask, "egfr_class"]    = "Unknown"
        merged.loc[unknown_mask, "cancer_type"]   = "Unknown"
        merged.loc[unknown_mask, "protein_change"] = ""
        merged.loc[unknown_mask, "driver_category"] = ""

    merged["egfr_driver"]  = merged["egfr_driver"].astype(int)
    merged["egfr_mutated"] = merged["egfr_mutated"].astype(int)

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    slides     = build_slide_inventory()
    driver_df, other_df, wt_df = load_mutation_data()
    merged     = merge_labels(slides, driver_df, other_df, wt_df)

    # Reorder columns
    cols = [
        "patient", "slide", "gcs_path", "source_bucket",
        "sample_id", "sample_type_code", "sample_type", "slide_type",
        "cancer_type",
        "egfr_driver", "egfr_mutated",
        "protein_change", "driver_category", "egfr_class",
    ]
    merged = merged[cols].sort_values(["patient", "slide"]).reset_index(drop=True)

    # Preserve leading zeros in sample_type_code (01, 11, etc.)
    merged["sample_type_code"] = merged["sample_type_code"].astype(str).str.zfill(2)

    # Deduplicate: same SVS filename may appear in multiple buckets
    # Priority: TCGA_LUAD_SVS > EGFR_SVS > egfr_exon19_luad
    bucket_priority = {"TCGA_LUAD_SVS": 0, "EGFR_SVS": 1, "egfr_exon19_luad": 2}
    merged["_bucket_priority"] = merged["source_bucket"].map(bucket_priority).fillna(9)
    merged = (
        merged.sort_values("_bucket_priority")
        .drop_duplicates(subset="slide", keep="first")
        .drop(columns="_bucket_priority")
    )
    n_dup = 1110 - len(merged)
    if n_dup:
        print(f"  Deduplicated {n_dup} slides present in multiple buckets")

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows → {OUT_CSV}")

    # ---- Summary ----
    print("\n=== Annotation Summary ===")
    print(f"Total TCGA slides:        {len(merged)}")
    print(f"Unique patients:          {merged['patient'].nunique()}")
    print()

    tumor_slides = merged[merged["sample_type_code"] == "01"]
    print(f"Primary Tumor slides:     {len(tumor_slides)}")
    print(f"  EGFR Driver (label=1):  {(tumor_slides['egfr_driver'] == 1).sum()}")
    print(f"  EGFR Other  (mutated):  {((tumor_slides['egfr_mutated'] == 1) & (tumor_slides['egfr_driver'] == 0)).sum()}")
    print(f"  Wild-Type   (label=0):  {(tumor_slides['egfr_driver'] == 0).sum()}")
    print(f"  Unknown:                {(tumor_slides['egfr_driver'] == -1).sum()}")
    print()

    print("Slide type breakdown (Primary Tumor):")
    print(tumor_slides.groupby(["slide_type", "egfr_driver"]).size().unstack(fill_value=0))
    print()

    print("Cancer type breakdown:")
    print(merged.groupby(["cancer_type", "egfr_driver"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
