# scripts/export_tables.py
import argparse, pandas as pd, os
ap = argparse.ArgumentParser()
ap.add_argument("--plot_index", default="results/plots_smartgencon/plot_index.csv")
ap.add_argument("--out", default="results/tables_smartgencon")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
df = pd.read_csv(args.plot_index)

met = df[df["model"].str.contains(r"\[metrics\]", na=False)].copy()
met["Model"] = met["model"].str.replace(r" \[metrics\]", "", regex=True)

keep = ["split","Model","auc","pr_auc","ece","brier"]
out = met[keep].sort_values(["split","Model"]).reset_index(drop=True)

out.to_csv(f"{args.out}/metrics_overview.csv", index=False)
with open(f"{args.out}/metrics_overview.tex","w") as f:
    f.write(out.rename(columns={
        "split":"Split","auc":"AUC","pr_auc":"PR-AUC","ece":"ECE","brier":"Brier"
    }).to_latex(index=False, escape=False, float_format="%.3f"))

print("Saved:", f"{args.out}/metrics_overview.csv", "and", f"{args.out}/metrics_overview.tex")
