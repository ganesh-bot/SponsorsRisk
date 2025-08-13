# scripts/explainability/export_shap_gru9_per_trial.py
"""
SHAP GradientExplainer for GRU (numeric-only 9F model)
Exports sponsor-level per-trial SHAP summaries.
"""

import argparse, pathlib
import torch
import shap
import numpy as np
import pandas as pd
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
        out = self.model(x, lengths)
        return out.unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/sponsorsrisk_gru.pt")
    ap.add_argument("--seq_path", default="data/processed/sequences_3f.pkl")
    ap.add_argument("--out_csv", default="results/shap_gru9_per_trial.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[load] sequence data...")
    X, y, meta = load_sequence_data(args.seq_path)
    sponsor_names = meta.get("sponsor_names", ["?"] * len(y))
    trial_ids = meta.get("trial_ids", [f"trial_{i}" for i in range(len(y))])
    feature_names = meta.get("feature_names", [f"f{i}" for i in range(X.shape[2])])
    print("[debug] using feature_names:", feature_names)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    lengths = torch.tensor([np.count_nonzero(x[:, 0]) for x in X], dtype=torch.int64)

    valid_mask = lengths > 0
    X_tensor = X_tensor[valid_mask].to(device)
    lengths = lengths[valid_mask].to(device)
    y = np.array(y)[valid_mask.cpu().numpy()]
    sponsor_names = np.array(sponsor_names)[valid_mask.cpu().numpy()]
    trial_ids = np.array(trial_ids)[valid_mask.cpu().numpy()]

    print(f"[debug] valid samples: {X_tensor.shape[0]} | feature dim: {X_tensor.shape[-1]}")

    # Sample subset
    N = min(1000, X_tensor.shape[0])
    idx = np.random.choice(np.arange(X_tensor.shape[0]), size=N, replace=False)
    X_test = X_tensor[idx].detach().clone().requires_grad_()
    y_test = y[idx]
    sponsor_test = sponsor_names[idx]
    trial_test = trial_ids[idx]

    print("[load] model...")
    base_model = SponsorRiskGRU(input_dim=9, hidden_dim=64, num_layers=1).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))
    base_model.train()

    wrapped_model = GRUWrapper(base_model)
    wrapped_model.train()

    # SHAP
    print("[shap] computing...")
    explainer = shap.GradientExplainer(wrapped_model, X_test)
    shap_vals = explainer.shap_values(X_test)[0]  # shape: (N, 9)
    print("[shap] done | shape:", shap_vals.shape)

    # Handle shape: (N, 9, 1) → (N, 9)
    if shap_vals.ndim == 3 and shap_vals.shape[2] == 1:
        mean_shap = shap_vals.squeeze(-1)
    elif shap_vals.ndim == 2 and shap_vals.shape[1] == len(feature_names):
        mean_shap = shap_vals
    else:
        raise ValueError(f"[error] Unexpected SHAP shape: {shap_vals.shape}")


    with torch.no_grad():
        pred_probs = wrapped_model(X_test).squeeze().cpu().numpy()

    # Resize everything to match mean_shap shape (N=10)
    sponsor_test = sponsor_test[:mean_shap.shape[0]]
    trial_test = trial_test[:mean_shap.shape[0]]
    y_test = y_test[:mean_shap.shape[0]]
    pred_probs = pred_probs[:mean_shap.shape[0]]


    print("[debug] final row count:", mean_shap.shape[0])
    print("[debug] sponsor_test len:", len(sponsor_test))
    print("[debug] pred_probs len:", len(pred_probs))

    df = pd.DataFrame(mean_shap, columns=feature_names)
    df.insert(0, "pred_prob", pred_probs)
    df.insert(0, "label", y_test)
    df.insert(0, "trial_id", trial_test)
    df.insert(0, "sponsor", sponsor_test)

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("✅ Saved:", args.out_csv)


if __name__ == "__main__":
    main()
