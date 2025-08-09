import argparse
import json
import pandas as pd
from collections import defaultdict

from src.inference.pipeline import load_bundle, predict_from_sponsor_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/examples/sponsor_samples.csv")
    ap.add_argument("--out", default="results/batch_predictions.csv")
    args = ap.parse_args()

    model, calibrator, thresholds, vocab_maps, meta = load_bundle(models_dir="models")

    df = pd.read_csv(args.csv, parse_dates=["start_date"])
    df = df.sort_values(["sponsor_name", "start_date"])

    rows = []
    for sponsor, g in df.groupby("sponsor_name"):
        # Use entire history as of last row, predict risk for the "next" trial
        result = predict_from_sponsor_df(g, model, calibrator, thresholds, vocab_maps)
        rows.append({
            "sponsor_name": sponsor,
            "trials_in_history": len(g),
            "prob_calibrated": result["prob_calibrated"],
            "label": result["label"],
            "threshold_used": result["threshold_used"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"✅ Saved {args.out}")
    print(out_df)

if __name__ == "__main__":
    main()
