# scripts/explainability/export_shap_combined_debug.py
# SHAP explainability for CombinedGRU using numerical inputs only

import argparse, torch, shap, pandas as pd, numpy as np
from src.models.combined import CombinedGRU
from src.features.prepare_sequences import build_sequences_with_cats_trends
import torch.nn as nn


torch.backends.cudnn.enabled = False  # safer for SHAP


class XnOnlyWrapper(nn.Module):
    def __init__(self, model, Xc_fixed):
        super().__init__()
        self.model = model
        self.Xc_fixed = Xc_fixed

    def forward(self, Xn):
        B, T, _ = Xn.shape
        Xc = self.Xc_fixed.expand(B, -1, -1)
        lengths = torch.full((B,), T, dtype=torch.long)

        cat_keys = ["allocation", "masking", "primary_purpose", "intv_type"]
        emb_list = [self.model.embeds[key](Xc[:, :, i]) for i, key in enumerate(cat_keys)]
        X = torch.cat([Xn] + emb_list, dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.model.gru(packed)
        last = h[-1]
        logits = self.model.head(self.model.dropout(last)).squeeze(1)
        probs = torch.sigmoid(logits)
        return torch.stack([1 - probs, probs], dim=1)  # [B, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cpu")

    print("[load] sequence data...")
    out = build_sequences_with_cats_trends(args.csv_path, max_seq_len=10, verbose=True)
    Xn_all, Xc_all, y_all, L_all, sponsors, vocab_sizes, vocab_maps = out[:7]
    print(f"[debug] valid samples: {Xn_all.shape[0]} | feature dim: {Xn_all.shape[2]}")

    N_SHAP = 50
    idx = np.random.choice(Xn_all.shape[0], size=N_SHAP, replace=False)
    Xn_sample = Xn_all[idx].to(device)
    Xc_sample = Xc_all[idx].to(device)
    y_sample = y_all[idx]
    sponsors_sample = np.array(sponsors)[idx]

    # background: first 20 numerical samples, categorical average
    Xn_bg = Xn_sample[:20].requires_grad_()
    Xc_fixed = Xc_sample[:20].float().mean(dim=0, keepdim=True).long()

    print("[load] model...")
    model = CombinedGRU(num_dim=9, cat_vocab_sizes=vocab_sizes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    wrapper = XnOnlyWrapper(model, Xc_fixed.to(device)).to(device)

    print("[shap] computing on", Xn_sample.shape[0], "samples...")
    explainer = shap.GradientExplainer(wrapper, Xn_bg)
    shap_vals = explainer.shap_values(Xn_sample, nsamples=Xn_sample.shape[0])

    # shap_vals is a list of [class_0, class_1] shap arrays
    shap_arr = shap_vals[1]  # focus on class 1
    print("[debug] shap_arr shape:", shap_arr.shape)  # [N, T, D]

    # Reduce over time
    N_EFFECTIVE = shap_arr.shape[0]
    mean_feat = Xn_sample[:N_EFFECTIVE].detach().cpu().numpy().mean(axis=1)
    mean_shap = shap_arr.mean(axis=1)
    preds = wrapper(Xn_sample[:N_EFFECTIVE].to(device)).detach().cpu().numpy()[:, 1]



    print("[debug] mean_feat shape:", mean_feat.shape)
    print("[debug] mean_shap shape:", mean_shap.shape)
    print("[debug] preds shape:", preds.shape)

    feature_names = [
        "phase_enc", "enroll_z", "gap_months", "prior_fail_rate_last3",
        "enroll_delta_last3", "enroll_slope_last3", "phase_prog_last3",
        "gap_mean_last3", "intv_diversity_last5"
    ]

    df = pd.DataFrame(mean_feat, columns=feature_names)
    for i, name in enumerate(feature_names):
        df[f"shap_{name}"] = mean_shap[:, i]

    df.insert(0, "pred_prob", preds)
    df.insert(0, "label", y_sample[:N_EFFECTIVE].cpu().numpy())
    df.insert(0, "sponsor", sponsors_sample[:N_EFFECTIVE])


    df.to_csv(args.out_csv, index=False)
    print("✅ Saved:", args.out_csv)


if __name__ == "__main__":
    main()
