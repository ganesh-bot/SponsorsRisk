#!/usr/bin/env python
import argparse, pathlib
import numpy as np
import pandas as pd
import shap
import torch

from src.features.prepare_sequences import load_sequence_data
from src.models.gru import SponsorRiskGRU

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/sponsorsrisk_gru.pt")
    ap.add_argument("--seq_path", default="data/processed/sequences_3f.pkl")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    print("[load] loading sequences...")
    X, y, meta = load_sequence_data(args.seq_path)  # (N, T, 9), int labels, and meta["feature_names"]
    assert X.ndim == 3 and X.shape[2] == 9, f"Expected (N,T,9) but got {X.shape}"
    fnames = meta.get("feature_names", [f"f{i}" for i in range(X.shape[2])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] loading model from {args.model_path}")
    model = SponsorRiskGRU(input_dim=9, hidden_dim=64, num_layers=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Sample small background and evaluation set
    np.random.seed(42)
    N = len(X)
    bg_idx = np.random.choice(N, size=min(500, N), replace=False)
    test_idx = np.random.choice(N, size=min(2000, N), replace=False)
    X_bg = torch.tensor(X[bg_idx], dtype=torch.float32).to(device)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)

    def model_fn(x):
        # x: (N, T, 9) as numpy → Tensor → logits
        xt = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            p = model(xt)
        return p.cpu().numpy()

    print("[shap] running DeepExplainer...")
    explainer = shap.DeepExplainer(model_fn, X_bg)
    shap_vals = explainer.shap_values(X_test)[0]  # (N, T, 9)

    # Aggregate importance: mean |SHAP| over N and T per feature
    mean_abs = np.mean(np.abs(shap_vals), axis=(0,1))  # shape: (9,)
    df_imp = pd.DataFrame({"feature": fnames, "mean_abs_shap": mean_abs})
    df_imp.sort_values("mean_abs_shap", ascending=False).to_csv("results/shap_gru9_importance.csv", index=False)

    # Plot top-K
    top = df_imp.sort_values("mean_abs_shap", ascending=False).head(args.topk)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.title("GRU-9ch SHAP (mean |SHAP| across time)")
    plt.tight_layout()
    plt.savefig(outdir / "shap_gru9_top.png", dpi=200)
    print("Saved:", outdir / "shap_gru9_top.png")

if __name__ == "__main__":
    main()
