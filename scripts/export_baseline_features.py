# scripts/export_baseline_features.py
import pathlib, pandas as pd, numpy as np
from run_baseline import build_baseline_samples

OUT = pathlib.Path("results"); OUT.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    # 7F = numeric + one-hot + trends
    X, y, groups = build_baseline_samples(
        df, max_hist=10, use_categoricals=True, with_trends=True, verbose=True
    )
    # prefix for clarity
    X = X.add_prefix("X_")
    X["y"] = y.astype(int)
    X["sponsor"] = groups
    out = OUT / "baseline_7f_features.csv"
    X.to_csv(out, index=False)
    print(f"[ok] wrote {out} with shape {X.shape}")

if __name__ == "__main__":
    main()
