import argparse, pathlib
import torch
import shap
import numpy as np
import pandas as pd
from src.features.prepare_sequences import load_sequence_data
from src.models.gru import SponsorRiskGRU

torch.backends.cudnn.enabled = False  # Disable cuDNN for GRU backward pass

class GRUWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            mask = (x.abs().sum(dim=2) > 0).int()
            lengths = mask.sum(dim=1).cpu()
        out = self.model(x, lengths)
        return out.unsqueeze(1)  # shape [N, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/sponsorsrisk_gru.pt")
    ap.add_argument("--seq_path", default="data/processed/sequences_3f.pkl")
    ap.add_argument("--out_csv", default="results/shap_gru9_full.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[load] sequence data...")
    X, y, meta = load_sequence_data(args.seq_path)
    sponsor_names = np.array(meta.get("sponsor_names", ["?"] * len(y)))
    trial_ids = np.array(meta.get("trial_ids", [f"trial_{i}" for i in range(len(y))]))
    feature_names = meta.get("feature_names", [f"f{i}" for i in range(X.shape[2])])
    print("[debug] using feature_names:", feature_names)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    lengths = torch.tensor([np.count_nonzero(x[:, 0]) for x in X], dtype=torch.int64)
    valid_mask = lengths > 0

    X_tensor = X_tensor[valid_mask].to(device)
    lengths = lengths[valid_mask].to(device)

    y = np.array(y)[valid_mask.cpu().numpy()]
    sponsor_names = sponsor_names[valid_mask.cpu().numpy()]
    trial_ids = trial_ids[valid_mask.cpu().numpy()]

    print(f"[debug] valid samples: {X_tensor.shape[0]} | feature dim: {X_tensor.shape[-1]}")
    print(f"[debug] sponsor_names[0:5]: {sponsor_names[:5]}")

    N = min(1000, X_tensor.shape[0])
    idx = np.random.choice(np.arange(X_tensor.shape[0]), size=N, replace=False)
    X_test = X_tensor[idx].detach().clone().requires_grad_()
    y_test = y[idx]
    sponsor_test = sponsor_names[idx]
    trial_test = trial_ids[idx]
    features_test = X_test.detach().cpu().numpy()

    print("[load] model...")
    base_model = SponsorRiskGRU(input_dim=9, hidden_dim=64, num_layers=1).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))
    base_model.train()

    wrapped_model = GRUWrapper(base_model)
    wrapped_model.train()

    print("[shap] computing...")
    explainer = shap.GradientExplainer(wrapped_model, X_test)

    batch_shap_vals = []
    BATCH_SIZE = 100
    for i in range(0, X_test.shape[0], BATCH_SIZE):
        batch = X_test[i:i + BATCH_SIZE]
        svals = explainer.shap_values(
            batch,
            ranked_outputs=1,
            output_rank_order="max"
        )[0]
        svals = svals.squeeze(-1)       # (B, T, F)
        svals = svals.mean(axis=1)      # (B, F)
        batch_shap_vals.append(svals)

    mean_shap = np.concatenate(batch_shap_vals, axis=0)
    mean_feat = features_test.mean(axis=1)

    with torch.no_grad():
        pred_probs = wrapped_model(X_test).squeeze().cpu().numpy()

    df = pd.DataFrame(mean_feat, columns=[f"feat_{f}" for f in feature_names])
    for i, f in enumerate(feature_names):
        df[f"shap_{f}"] = mean_shap[:, i]

    df.insert(0, "pred_prob", pred_probs)
    df.insert(0, "label", y_test)
    df.insert(0, "trial_id", trial_test)
    df.insert(0, "sponsor", sponsor_test)

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("✅ Saved:", args.out_csv)
    print("[debug] unique sponsors in sample:", set(sponsor_test))


if __name__ == "__main__":
    main()
