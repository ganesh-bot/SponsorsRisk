# src/utils_data.py
import pandas as pd

VALID_PHASES = {
    "Phase 1", "Phase 1/Phase 2",
    "Phase 2", "Phase 2/Phase 3",
    "Phase 3"
}

FAIL_STATUSES = {"terminated", "withdrawn", "suspended"}
SUCCESS_STATUSES = {"completed"}

def norm_str(x):
    if pd.isna(x):
        return "unknown"
    return str(x).strip()

def norm_lower(x):
    return norm_str(x).lower()

def apply_cohort_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Interventional, valid phases, and keep only rows with keepable status."""
    df = df.copy()

    # normalize key fields
    df["study_type"] = df["study_type"].apply(norm_str)
    df["phase"] = df["phase"].apply(norm_str)
    df["overall_status"] = df["overall_status"].apply(norm_lower)

    # cohort: Interventional, valid phases
    df = df[df["study_type"] == "Interventional"]
    df = df[df["phase"].isin(VALID_PHASES)]

    # keepable statuses only (success/fail); drop everything else
    keep = df["overall_status"].isin(FAIL_STATUSES | SUCCESS_STATUSES)
    df = df[keep].reset_index(drop=True)
    return df

def label_from_status(status: str) -> int:
    """Return 1 = fail, 0 = success (assumes status already normalized to lower)."""
    if status in FAIL_STATUSES:
        return 1
    if status in SUCCESS_STATUSES:
        return 0
    # should not happen if filtered, default to 0
    return 0
