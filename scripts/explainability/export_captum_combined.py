# scripts/explainability/export_captum_combined.py
# Captum-based interpretability for CombinedGRU model (numerical-only)

import argparse, torch, pandas as pd, numpy as np
from captum.attr import IntegratedGradients
from src.models.combined import CombinedGRU
from src.features.prepare_sequences import build_sequences_with_cats_trends
import torch.nn as nn


torch.backends.cudnn.enabled = False  # safer for interpretability

class XnOnlyWrapper(nn.Module):
    def __init__(self, model, Xc_fixed):
        super().__init__()
        self.model = model
        self.Xc_fixed = Xc_fixed

    def forward(self, Xn):
        B = Xn.size(0)
        Xc = self.Xc_fixed.expand(B, -1, -1)
        lengths = torch.full((B,), Xn.size(1), dtype=torch.long)

        cat_keys = ["allocation", "masking", "primary_purpose", "intv_type"]
        emb_list = [self.model.embeds[key](Xc[:, :, i]) for i, key in enumerate(cat_keys)]

        X = torch.cat([Xn] + emb_list, dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.model.gru(packed)
        last = h[-1]
        logits = self.model.head(self.model.dropout(last)).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs  # single output

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

    N_SAMPLES = 50
    idx = np.random.choice(Xn_all.shape[0], size=N_SAMPLES, replace=False)
    Xn_sample, Xc_sample, y_sample, sponsors_sample = Xn_all[idx], Xc_all[idx], y_all[idx], np.array(sponsors)[idx]

    Xn_sample = Xn_sample.to(device).requires_grad_()
    Xc_fixed = Xc_sample[:20].float().mean(dim=0, keepdim=True).long().to(device)

    print("[load] model...")
    model = CombinedGRU(num_dim=9, cat_vocab_sizes=vocab_sizes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    wrapper = XnOnlyWrapper(model, Xc_fixed)
    ig = IntegratedGradients(wrapper)

    print("[captum] computing attributions...")
    baseline = torch.zeros_like(Xn_sample)
    attributions, _ = ig.attribute(Xn_sample, baselines=baseline, return_convergence_delta=True)
    print("[captum] done")

    # [B, T, D] -> mean over time
    mean_feat = Xn_sample.detach().cpu().numpy().mean(axis=1)   # [B, D]
    mean_attr = attributions.detach().cpu().numpy().mean(axis=1)
    preds = wrapper(Xn_sample).detach().cpu().numpy()

    feature_names = [
        "phase_enc", "enroll_z", "gap_months", "prior_fail_rate_last3",
        "enroll_delta_last3", "enroll_slope_last3", "phase_prog_last3",
        "gap_mean_last3", "intv_diversity_last5"
    ]

    df = pd.DataFrame(mean_feat, columns=feature_names)
    for i, name in enumerate(feature_names):
        df[f"attr_{name}"] = mean_attr[:, i]

    df.insert(0, "pred_prob", preds)
    df.insert(0, "label", y_sample.cpu().numpy())
    df.insert(0, "sponsor", sponsors_sample)

    df.to_csv(args.out_csv, index=False)
    print("✅ Saved:", args.out_csv)

if __name__ == "__main__":
    main()
