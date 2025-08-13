# scripts/diagnostics_shap_baseline.py
#!/usr/bin/env python
import argparse, pathlib
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

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

    # Preprocess: impute NaN→0, drop constant cols, center only (no / std)
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("const", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler(with_std=False)),  # center only to avoid div-by-zero
    ])
    Xs = pre.fit_transform(X)

    # Track surviving feature names after VarianceThreshold
    mask = pre.named_steps["const"].get_support()
    kept_names = np.array(Xcols)[mask]

    # Fit linear model
    lr = LogisticRegression(max_iter=300, class_weight="balanced")
    lr.fit(Xs, y)

    # SHAP for linear model on preprocessed data
    explainer = shap.LinearExplainer(lr, Xs, feature_names=kept_names)
    shap_vals = explainer.shap_values(Xs)

    imp = pd.DataFrame({
        "feature": kept_names,
        "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    imp.to_csv("results/shap_baseline_importance.csv", index=False)

    top = imp.head(args.topk)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.title("Baseline SHAP (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(outdir / "shap_baseline_top.png", dpi=200)

    # Quick health report
    n_nan = int(np.isnan(X).sum())
    print(f"[ok] saved: {outdir/'shap_baseline_top.png'} and results/shap_baseline_importance.csv")
    print(f"[health] NaNs in raw X: {n_nan} | kept features: {len(kept_names)}/{len(Xcols)}")

if __name__ == "__main__":
    main()
