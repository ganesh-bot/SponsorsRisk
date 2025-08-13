# scripts/dump_sequences_3f.py
"""
Dump 3F+trends sequence data for GRU input (9 numeric features).
Output: data/processed/sequences_3f.pkl
"""

import pathlib, pickle
import pandas as pd
from src.features.prepare_sequences import build_sequences_rich_trends

def main():
    csv_path = "data/aact_extracted.csv"
    outpath = pathlib.Path("data/processed/sequences_3f.pkl")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {csv_path}")
    X, y, lengths, sponsors = build_sequences_rich_trends(csv_path, max_seq_len=10, verbose=True)

    feature_names = [
        "phase_enc", "enroll_z", "gap_months",
        "prior_fail_rate_last3", "enroll_delta_last3", "enroll_slope_last3",
        "phase_prog_last3", "gap_mean_last3", "intv_diversity_last5"
    ]

    # ✅ Add sponsor_names and trial_ids to meta
    meta = {
        "feature_names": feature_names,
        "sponsor_names": sponsors,
        "trial_ids": [f"trial_{i}" for i in range(len(y))]
    }

    with open(outpath, "wb") as f:
        pickle.dump({
            "X": X.numpy(),
            "y": y.numpy(),
            "meta": meta
        }, f)

    print(f"[ok] wrote {outpath} | shape={X.shape} | positives={int(y.sum())}")

if __name__ == "__main__":
    main()
