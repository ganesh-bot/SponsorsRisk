import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve, RocCurveDisplay,
    PrecisionRecallDisplay, brier_score_loss
)
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.prepare_sequences import (
    build_sequences_rich_trends,
    build_sequences_with_cats_trends,
)
from run_baseline import build_baseline_samples
from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.model_combined import CombinedGRU
from src.train import infer_lengths_from_padding

import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = "results/plots"

# ---------------- helpers ----------------

def train_seq_get_probs(model, Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=None):
    """Train a seq model (GRU/Transformer/Combined wrapper) and return train/val probabilities."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
    tr_ds, va_ds = TensorDataset(Xtr, ytr), TensorDataset(Xva, yva)
    tr_ld = DataLoader(tr_ds, batch_size=batch, sampler=RandomSampler(tr_ds), pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch, sampler=SequentialSampler(va_ds), pin_memory=True)

    best_auc, best, pat, PATIENCE = -1.0, None, 0, 5
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            L = infer_lengths_from_padding(xb)
            logits = model(xb, L)
            loss = F.binary_cross_entropy_with_logits(
                logits, yb,
                pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # quick val pass for scheduler & patience using logloss
        model.eval(); allp, ally = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                L = infer_lengths_from_padding(xb)
                allp.append(torch.sigmoid(model(xb, L)).cpu()); ally.append(yb.cpu())
        y_true = torch.cat(ally).numpy(); p_va = torch.cat(allp).numpy()
        eps = 1e-7
        val_logloss = -np.mean(y_true*np.log(p_va+eps) + (1-y_true)*np.log(1-p_va+eps))
        sched.step(val_logloss)

        # simple auc tracking (threshold-free)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, p_va)
        except Exception:
            auc = -1.0

        if auc > best_auc:
            best_auc, best, pat = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best is not None:
        model.load_state_dict(best)

    def infer_probs(X, batch=512):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = X[i:i+batch].to(DEVICE)
                L = infer_lengths_from_padding(xb)
                out.append(torch.sigmoid(model(xb, L)).cpu())
        return torch.cat(out).numpy()

    p_tr = infer_probs(Xtr)
    p_va = infer_probs(Xva)
    return p_tr, p_va

def calibrate_isotonic(y_tr, p_tr, p_va):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_tr, y_tr)
    return ir.transform(p_va)

def plot_all_curves(y_val, p_val, title_prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_val, p_val)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f"{title_prefix} - ROC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_ROC.png"), dpi=180)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_val, p_val)
    PrecisionRecallDisplay(precision=prec, recall=rec).plot()
    plt.title(f"{title_prefix} - PR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_PR.png"), dpi=180)
    plt.close()

    # Reliability (calibration curve)
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y_val, p_val, n_bins=10, ax=ax)
    bs = brier_score_loss(y_val, p_val)
    ax.set_title(f"{title_prefix} - Reliability (Brier={bs:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{title_prefix}_Reliability.png"), dpi=180)
    plt.close()

# ---------------- main ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_baseline", action="store_true", help="Also plot for Baseline-7F+trends")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    # ---------------- Baseline (optional) ----------------
    rows = []

    if args.include_baseline:
        df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
        Xb, yb, groups = build_baseline_samples(
            df, max_hist=10, use_categoricals=True, with_trends=True, verbose=False
        )
        Xtr, Xva = Xb.iloc[tr_idx], Xb.iloc[va_idx]
        ytr, yva = yb[tr_idx], yb[va_idx]

        lr = LogisticRegression(max_iter=500, class_weight="balanced")
        lr.fit(Xtr, ytr)
        p_tr = lr.predict_proba(Xtr)[:, 1]
        p_va = lr.predict_proba(Xva)[:, 1]

        # uncalibrated plots
        plot_all_curves(yva, p_va, "Baseline7F_trends_uncal", PLOTS_DIR)

        # isotonic calibration & plots
        p_va_iso = calibrate_isotonic(ytr, p_tr, p_va)
        plot_all_curves(yva, p_va_iso, "Baseline7F_trends_iso", PLOTS_DIR)

        rows.append({"model":"Baseline7F_trends_uncal", "source":"val_probs"})
        rows.append({"model":"Baseline7F_trends_iso",   "source":"val_probs_iso"})

    # ---------------- GRU (9ch) ----------------
    X, y, L, _ = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
    Xtr, Xva = X[tr_idx], X[va_idx]; ytr, yva = y[tr_idx], y[va_idx]
    pos_w = torch.tensor((ytr==0).sum().item()/max((ytr==1).sum().item(),1), dtype=torch.float32) * 1.2

    gru = SponsorRiskGRU(input_dim=Xtr.shape[2], hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)
    p_tr, p_va = train_seq_get_probs(gru, Xtr, ytr, Xva, yva, epochs=args.epochs, lr=1e-3, batch=args.batch, pos_weight=pos_w)
    plot_all_curves(yva.numpy(), p_va, "GRU9_uncal", PLOTS_DIR)
    p_va_iso = calibrate_isotonic(ytr.numpy(), p_tr, p_va)
    plot_all_curves(yva.numpy(), p_va_iso, "GRU9_iso", PLOTS_DIR)
    rows.append({"model":"GRU9_uncal", "source":"val_probs"})
    rows.append({"model":"GRU9_iso",   "source":"val_probs_iso"})

    # ---------------- Transformer (9ch) ----------------
    tx = SponsorRiskTransformer(input_dim=Xtr.shape[2], d_model=64, nhead=4, num_layers=2,
                                dim_feedforward=128, dropout=0.1).to(DEVICE)
    p_tr, p_va = train_seq_get_probs(tx, Xtr, ytr, Xva, yva, epochs=args.epochs, lr=1e-3, batch=args.batch, pos_weight=pos_w)
    plot_all_curves(yva.numpy(), p_va, "Transformer9_uncal", PLOTS_DIR)
    p_va_iso = calibrate_isotonic(ytr.numpy(), p_tr, p_va)
    plot_all_curves(yva.numpy(), p_va_iso, "Transformer9_iso", PLOTS_DIR)
    rows.append({"model":"Transformer9_uncal", "source":"val_probs"})
    rows.append({"model":"Transformer9_iso",   "source":"val_probs_iso"})

    # ---------------- Combined (9+4) ----------------
    Xn, Xc, yC, Lc, sponsorsC, vocab_sizes = build_sequences_with_cats_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
    Xn_tr, Xn_va = Xn[tr_idx], Xn[va_idx]
    Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
    yC_tr, yC_va = yC[tr_idx], yC[va_idx]
    L_tr, L_va   = Lc[tr_idx], Lc[va_idx]
    pos_w2 = torch.tensor((yC_tr==0).sum().item()/max((yC_tr==1).sum().item(),1), dtype=torch.float32) * 1.2

    cmb = CombinedGRU(num_dim=Xn_tr.shape[2], cat_vocab_sizes=vocab_sizes,
                      emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)
    # wrap combined into the same training util by adapting signature:
    def train_combined(cmb_model, Xn_tr, Xc_tr, y_tr, L_tr, Xn_va, Xc_va, y_va, L_va, epochs, lr, batch, pos_weight):
        opt = torch.optim.Adam(cmb_model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
        tr_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, y_tr)
        va_ds = TensorDataset(Xn_va, Xc_va, L_va, y_va)
        tr_ld = DataLoader(tr_ds, batch_size=batch, sampler=RandomSampler(tr_ds), pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=batch, sampler=SequentialSampler(va_ds), pin_memory=True)

        best_auc, best, pat, PATIENCE = -1.0, None, 0, 5
        for ep in range(1, epochs+1):
            cmb_model.train()
            for xn, xc, l, yb in tr_ld:
                xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
                logits = cmb_model(xn, xc, l)
                loss = F.binary_cross_entropy_with_logits(
                    logits, yb,
                    pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
                )
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(cmb_model.parameters(), max_norm=1.0)
                opt.step()

            # val
            cmb_model.eval(); allp, ally = [], []
            with torch.no_grad():
                for xn, xc, l, yb in va_ld:
                    xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
                    allp.append(torch.sigmoid(cmb_model(xn, xc, l)).cpu()); ally.append(yb.cpu())
            y_true = torch.cat(ally).numpy(); p_va = torch.cat(allp).numpy()
            eps = 1e-7
            val_logloss = -np.mean(y_true*np.log(p_va+eps) + (1-y_true)*np.log(1-p_va+eps))
            sched.step(val_logloss)

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, p_va)
            except Exception:
                auc = -1.0

            if auc > best_auc:
                best_auc, best, pat = auc, {k:v.cpu().clone() for k,v in cmb_model.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        if best is not None:
            cmb_model.load_state_dict(best)

        def infer_probs(Xn, Xc, L, batch=512):
            cmb_model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(Xn), batch):
                    xn, xc, l = Xn[i:i+batch].to(DEVICE), Xc[i:i+batch].to(DEVICE), L[i:i+batch].to(DEVICE)
                    out.append(torch.sigmoid(cmb_model(xn, xc, l)).cpu())
            return torch.cat(out).numpy()

        p_tr = infer_probs(Xn_tr, Xc_tr, L_tr)
        p_va = infer_probs(Xn_va, Xc_va, L_va)
        return p_tr, p_va

    p_tr, p_va = train_combined(cmb, Xn_tr, Xc_tr, yC_tr, L_tr, Xn_va, Xc_va, yC_va, L_va,
                                epochs=args.epochs, lr=1e-3, batch=args.batch, pos_weight=pos_w2)
    plot_all_curves(yC_va.numpy(), p_va, "Combined9p4_uncal", PLOTS_DIR)
    p_va_iso = calibrate_isotonic(yC_tr.numpy(), p_tr, p_va)
    plot_all_curves(yC_va.numpy(), p_va_iso, "Combined9p4_iso", PLOTS_DIR)
    rows.append({"model":"Combined9p4_uncal", "source":"val_probs"})
    rows.append({"model":"Combined9p4_iso",   "source":"val_probs_iso"})

    # summary index
    pd.DataFrame(rows).to_csv(os.path.join(PLOTS_DIR, "plot_index.csv"), index=False)
    print(f"✅ Saved ROC/PR/Reliability plots under {PLOTS_DIR}")
