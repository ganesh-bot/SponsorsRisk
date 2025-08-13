# scripts/explainability/plot_waterfall_gru9.py

import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt

def plot_waterfall_for_trial(df, sponsor_name, trial_id=None, save_path=None):
    df_sponsor = df[df["sponsor"] == sponsor_name]

    if df_sponsor.empty:
        raise ValueError(f"No trials found for sponsor: {sponsor_name}")

    if trial_id:
        row = df_sponsor[df_sponsor["trial_id"] == trial_id].iloc[0]
    else:
        # default: pick trial with highest predicted risk
        row = df_sponsor.sort_values("pred_prob", ascending=False).iloc[0]

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    shap_cols = [c for c in df.columns if c.startswith("shap_")]

    base_value = row["pred_prob"] - row[shap_cols].sum()
    shap_values = row[shap_cols].values
    features = row[feat_cols].values

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=features,
            feature_names=[c.replace("feat_", "") for c in feat_cols]
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall\nSponsor: {sponsor_name}\nTrial: {row['trial_id']}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/shap_gru9_full.csv")
    ap.add_argument("--sponsor", type=str, required=True)
    ap.add_argument("--trial_id", type=str, default=None)
    ap.add_argument("--save_path", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plot_waterfall_for_trial(df, args.sponsor, args.trial_id, args.save_path)

if __name__ == "__main__":
    main()
