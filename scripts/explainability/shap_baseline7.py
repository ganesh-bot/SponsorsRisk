# scripts/explainability/shap_baseline7.py
"""
Compute SHAP for Baseline Logistic Regression (7F+trends).
Outputs:
- results/plots/shap_baseline7_top.png
- results/shap_baseline7_importance.csv
"""

import argparse, pathlib
import numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="results/baseline_7f_features.csv")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    if "y" not in df.columns:
        raise SystemExit("CSV must contain a 'y' column.")
    Xcols = [c for c in df.columns if c.startswith("X_")]
    X = df[Xcols].replace([np.inf, -np.inf], np.nan).values
    y = df["y"].astype(int).values

    # Preprocess: impute NaNs, drop constant columns, standardize
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("const", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler(with_std=False)),
    ])
    Xs = pre.fit_transform(X)
    kept = np.array(Xcols)[pre.named_steps["const"].get_support()]

    lr = LogisticRegression(max_iter=300, class_weight="balanced")
    lr.fit(Xs, y)

    explainer = shap.LinearExplainer(lr, Xs, feature_names=kept)
    shap_vals = explainer.shap_values(Xs)

    imp = pd.DataFrame({"feature": kept, "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0)})
    imp.sort_values("mean_abs_shap", ascending=False)\
       .to_csv("results/shap_baseline7_importance.csv", index=False)

    top = imp.head(args.topk)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.title("Baseline SHAP (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(outdir / "shap_baseline7_top.png", dpi=200)
    print("Saved:", outdir / "shap_baseline7_top.png")

if __name__ == "__main__":
    main()
