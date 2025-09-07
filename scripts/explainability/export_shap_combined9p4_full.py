import argparse, pathlib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from src.models.combined import CombinedGRU
from src.features.prepare_sequences import build_sequences_with_cats_trends

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CombinedWrapper(nn.Module):
    def __init__(self, model, lengths):
        super().__init__()
        self.model = model
        self.lengths = lengths

    def forward(self, inputs):
        Xn, Xc = inputs
        return self.model(Xn, Xc, self.lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--nsamples", type=int, default=1000)
    args = parser.parse_args()

    # === Load Data ===
    print("[load] sequence data...")
    out = build_sequences_with_cats_trends(args.csv_path, max_seq_len=10, verbose=False)
    Xn_all, Xc_all, y_all, L_all, sponsors, vocab_sizes, vocab_maps = out

    # Generate placeholders for fields not returned
    trial_ids = [f"trial_{i}" for i in range(len(y_all))]
    feature_names = [f"feat_{i}" for i in range(Xn_all.shape[2])]
    cat_names = [f"cat_{i}" for i in range(Xc_all.shape[2])]


    N = min(args.nsamples, Xn_all.shape[0])
    Xn = Xn_all[:N].to(device)
    Xc = Xc_all[:N].to(device)
    y = y_all[:N].cpu().numpy()
    L = L_all[:N].to(device)
    sponsors = sponsors[:N]
    trial_ids = trial_ids[:N]

    print(f"[debug] valid samples: {N} | feature dim: {Xn.shape[-1]}")
    print(f"[debug] Xn shape: {Xn.shape} | Xc shape: {Xc.shape} | L shape: {L.shape}")

    # === Load Model ===
    print("[load] model...")
    model = CombinedGRU(num_dim=Xn.shape[2], cat_vocab_sizes=vocab_sizes,
                        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    model_wrapper = CombinedWrapper(model, L)

    # === SHAP ===
    print("[shap] preparing background...")
    Xn_bg = Xn[:100]
    Xc_bg = Xc[:100]
    background = (Xn_bg, Xc_bg)
    print(f"[debug] Background Xn shape: {Xn_bg.shape} | Xc shape: {Xc_bg.shape}")

    print("[shap] creating explainer...")
    explainer = shap.GradientExplainer(model_wrapper, background)

    print("[shap] computing SHAP values...")
    shap_raw = explainer.shap_values((Xn, Xc))  # returns [Xn_shap, Xc_shap]
    shap_num, shap_cat = shap_raw
    print(f"[shap] SHAP output shapes: numeric={shap_num.shape}, categorical={shap_cat.shape}")

    # === Aggregate over time ===
    mean_shap_num = shap_num.mean(axis=1)  # (N, 9)
    mean_feat_num = Xn.detach().cpu().numpy().mean(axis=1)

    # === Model Predictions ===
    with torch.no_grad():
        probs = torch.sigmoid(model(Xn, Xc, L)).cpu().numpy()

    # === Build DataFrame ===
    df = pd.DataFrame(mean_feat_num, columns=[f"feat_{f}" for f in feature_names])
    for i, f in enumerate(feature_names):
        df[f"shap_{f}"] = mean_shap_num[:, i]

    df.insert(0, "pred_prob", probs)
    df.insert(0, "label", y)
    df.insert(0, "trial_id", trial_ids)
    df.insert(0, "sponsor", sponsors)

    # === Save CSV ===
    outpath = pathlib.Path(args.out_csv)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    print("✅ Saved:", args.out_csv)


if __name__ == "__main__":
    main()
