"""
Analyze SHAP per-trial output for GRU model.
Generates CSV summaries and plots for failed vs successful trials.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_shap(csv_path="results/shap_gru9_per_trial.csv", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    
    shap_cols = df.columns[4:]  # skip sponsor, trial_id, label, pred_prob
    df_fail = df[df.label == 1.0]
    df_success = df[df.label == 0.0]
    
    print(f"[info] # failed trials: {len(df_fail)}, # successful trials: {len(df_success)}")

    # === Mean SHAPs ===
    mean_fail = df_fail[shap_cols].abs().mean().sort_values(ascending=False)
    mean_success = df_success[shap_cols].abs().mean().sort_values(ascending=False)
    
    mean_fail.to_csv(os.path.join(output_dir, "summary_shap_failed.csv"), header=["mean_abs_shap"])
    mean_success.to_csv(os.path.join(output_dir, "summary_shap_success.csv"), header=["mean_abs_shap"])
    
    # === Per-Sponsor Summary (Failed only) ===
    df_fail.groupby("sponsor")[shap_cols].mean().to_csv(os.path.join(output_dir, "shap_per_sponsor.csv"))

    # === Barplot Utility ===
    def plot_bar(series, title, fname):
        plt.figure(figsize=(8, 6))
        series.head(10).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Mean |SHAP| Value")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=200)
        print("✅ Saved:", fname)

    plot_bar(mean_fail, "Top SHAP Features – Failed Trials", "shap_top_failed.png")
    plot_bar(mean_success, "Top SHAP Features – Successful Trials", "shap_top_success.png")

    print("\n✅ Done analyzing SHAP results.\n")


if __name__ == "__main__":
    analyze_shap()
