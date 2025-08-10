# scripts/export_model.py
import argparse, os, json, joblib
import numpy as np
import torch
from src.train.metrics import compute_auc_pr, best_f1_threshold

from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.model_combined import CombinedGRU
from src.prepare_sequences import build_sequences_rich_trends, build_sequences_with_cats_trends
from sklearn.isotonic import IsotonicRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAP = {
    "gru": SponsorRiskGRU,
    "transformer": SponsorRiskTransformer,
    "combined": CombinedGRU
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_MAP.keys(), required=True)
    ap.add_argument("--weights", required=True, help="Path to trained .pt file")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load frozen split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    if args.model == "combined":
        Xn, Xc, y, L, sponsors, vocab_sizes, *extra = build_sequences_with_cats_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
        model = CombinedGRU(num_dim=Xn.shape[2], cat_vocab_sizes=vocab_sizes, emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1)
        tr_data = (Xn[tr_idx], Xc[tr_idx], L[tr_idx], y[tr_idx])
        va_data = (Xn[va_idx], Xc[va_idx], L[va_idx], y[va_idx])
        meta = {"num_dim": Xn.shape[2], "cat_vocab_sizes": vocab_sizes, "max_seq_len": Xn.shape[1]}
    elif args.model == "gru":
        X, y, lengths, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10)
        model = SponsorRiskGRU(input_dim=X.shape[2], hidden_dim=64, num_layers=1, dropout=0.1)
        tr_data = (X[tr_idx], lengths[tr_idx], y[tr_idx])
        va_data = (X[va_idx], lengths[va_idx], y[va_idx])
        meta = {"input_dim": X.shape[2], "max_seq_len": X.shape[1]}
    elif args.model == "transformer":
        X, y, lengths, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10)
        model = SponsorRiskTransformer(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
        tr_data = (X[tr_idx], lengths[tr_idx], y[tr_idx])
        va_data = (X[va_idx], lengths[va_idx], y[va_idx])
        meta = {"input_dim": X.shape[2], "max_seq_len": X.shape[1]}

    # Load weights
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Get train/val probs
    def get_probs(data):
        all_p, all_y = [], []
        with torch.no_grad():
            if args.model == "combined":
                Xn, Xc, L, yb = data
                for i in range(0, len(yb), 256):
                    xn_b = Xn[i:i+256].to(DEVICE)
                    xc_b = Xc[i:i+256].to(DEVICE)
                    l_b  = L[i:i+256].to(DEVICE)
                    p = torch.sigmoid(model(xn_b, xc_b, l_b))
                    all_p.append(p.cpu()); all_y.append(yb[i:i+256])
            else:
                X, L, yb = data
                for i in range(0, len(yb), 256):
                    xb = X[i:i+256].to(DEVICE)
                    l_b = L[i:i+256].to(DEVICE)
                    p = torch.sigmoid(model(xb, l_b))
                    all_p.append(p.cpu()); all_y.append(yb[i:i+256])
        return np.concatenate(all_p), np.concatenate(all_y)

    p_tr, y_tr = get_probs(tr_data)
    p_va, y_va = get_probs(va_data)

    # Fit isotonic calibrator
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    p_va_iso = iso.transform(p_va)

    # Save model + calibrator
    model_path = os.path.join(args.outdir, f"sponsorsrisk_{args.model}.pt")
    torch.save(model.state_dict(), model_path)
    joblib.dump(iso, os.path.join(args.outdir, "calibrator_isotonic.pkl"))

    # Save thresholds
    thr, acc_thr, f1_thr = best_f1_threshold(y_va, p_va_iso)
    with open(os.path.join(args.outdir, "thresholds.json"), "w") as f:
        json.dump({"threshold": thr, "best_f1": f1_thr}, f, indent=2)

    # Save meta
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Exported {args.model} to {args.outdir}")
    auc, pr = compute_auc_pr(y_va, p_va_iso)
    print(f"Val AUC (iso): {auc:.3f} | PR-AUC (iso): {pr:.3f} | Best-F1: {f1_thr:.3f} @ thr={thr:.3f}")

if __name__ == "__main__":
    main()
