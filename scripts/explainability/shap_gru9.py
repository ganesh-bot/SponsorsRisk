# scripts/explainability/shap_gru9.py

import argparse, pathlib
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features.prepare_sequences import load_sequence_data
from src.models.gru import SponsorRiskGRU

# ✅ Disable cuDNN to prevent backward crash with GRU
torch.backends.cudnn.enabled = False


class GRUWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            mask = (x.abs().sum(dim=2) > 0).int()
            lengths = mask.sum(dim=1).cpu()
        out = self.model(x, lengths)  # (N,)
        return out.unsqueeze(1)       # (N, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/sponsorsrisk_gru.pt")
    ap.add_argument("--seq_path", default="data/processed/sequences_3f.pkl")
    ap.add_argument("--outdir", default="results/plots")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[load] sequence data...")
    X, y, meta = load_sequence_data(args.seq_path)
    n_features = X.shape[2]
    feature_names = meta.get("feature_names", [])
    if not isinstance(feature_names, list) or len(feature_names) != n_features:
        feature_names = [f"f{i}" for i in range(n_features)]  # fallback

    print(f"[debug] using {len(feature_names)} feature names: {feature_names}")

    X_tensor_all = torch.tensor(X, dtype=torch.float32)
    lengths_all = torch.tensor([np.count_nonzero(x[:, 0]) for x in X], dtype=torch.int64)

    valid_mask = lengths_all > 0
    X_tensor = X_tensor_all[valid_mask].to(device)
    lengths = lengths_all[valid_mask].to(device)

    print(f"[debug] total={len(X)} | valid={len(X_tensor)} | shape={X_tensor.shape}")

    # Sample for SHAP
    bg_size = min(128, len(X_tensor))
    test_size = min(1000, len(X_tensor))
    np.random.seed(42)
    idx = np.random.permutation(len(X_tensor))
    bg_idx = idx[:bg_size]
    test_idx = idx[bg_size:bg_size + test_size]

    X_bg = X_tensor[bg_idx].detach().requires_grad_()
    X_test = X_tensor[test_idx].detach().requires_grad_()

    print("[load] model...")
    base_model = SponsorRiskGRU(input_dim=9, hidden_dim=64, num_layers=1).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))
    base_model.train()  # Needed to enable RNN backward

    wrapped_model = GRUWrapper(base_model)
    wrapped_model.train()  # Just to be extra safe

    # Run SHAP GradientExplainer
    # Run SHAP GradientExplainer
    print("[shap] computing...")
    explainer = shap.GradientExplainer(wrapped_model, X_bg)
    shap_vals_all = explainer.shap_values(X_test)
    shap_vals = shap_vals_all[0] if isinstance(shap_vals_all, list) else shap_vals_all

    # Handle 4D shape (N, T, F, 1)
    if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals.squeeze(-1)
    print("[shap] done")
    print("[debug] shap_vals shape:", shap_vals.shape)

    # Check shape
    if len(shap_vals.shape) != 3:
        raise ValueError(f"Unexpected SHAP shape: {shap_vals.shape}")

    # Aggregate feature importance over all time steps and samples
    mean_abs = np.mean(np.abs(shap_vals), axis=(0, 1))
    print("[debug] mean_abs shape:", mean_abs.shape)
    print("[debug] feature_names:", feature_names)

    if len(mean_abs) != len(feature_names):
        raise ValueError("Mismatch between SHAP values and feature names.")

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False)
    df.to_csv(outdir / "shap_gru9_importance.csv", index=False)

    # Plot top 20
    top = df.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.title("GRU SHAP (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(outdir / "shap_gru9_top.png", dpi=200)
    print("✅ Saved:", outdir / "shap_gru9_top.png")

if __name__ == "__main__":
    main()
