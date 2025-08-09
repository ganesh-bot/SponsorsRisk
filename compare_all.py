# compare_all.py
import os, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.isotonic import IsotonicRegression

from src.prepare_sequences import (
    build_sequences_rich_trends,          # -> X (N,T,9), y, L, sponsors
    build_sequences_with_cats_trends,     # -> X_num (N,T,9), X_cat (N,T,4), y, L, sponsors, vocab_sizes
)
from run_baseline import build_baseline_samples  # robust baseline builder w/ trend features

from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.model_combined import CombinedGRU
from src.train import infer_lengths_from_padding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Utilities
# ============================================================

def evaluate_all(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    try:
        prauc = average_precision_score(y_true, y_prob)
    except ValueError:
        prauc = float("nan")
    return acc, f1, auc, prauc

def best_f1_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    # thresholds len = len(p) - 1; align by prepending 0
    t = np.r_[0.0, t]
    f1 = 2 * (p * r) / np.maximum(p + r, 1e-9)
    k = np.nanargmax(f1)
    return float(t[k]), float(f1[k])

def calibrate_isotonic(y_tr, p_tr, p_va):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_tr, y_tr)
    return ir.transform(p_va)

def sponsor_type_map(df):
    """
    Map sponsor_name -> most common sponsor_type (normalized).
    Tries to find a suitable column when 'sponsor_type' is missing.
    """
    if "sponsor_type" not in df.columns:
        candidates = [c for c in df.columns if re.search(r"(sponsor|agency).*class|sponsor_type", c, re.I)]
        if candidates:
            df = df.rename(columns={candidates[0]: "sponsor_type"})
        else:
            df["sponsor_type"] = "unknown"
    st = (df.assign(sponsor_type=df["sponsor_type"].fillna("unknown").astype(str).str.strip().str.lower())
            .groupby("sponsor_name")["sponsor_type"]
            .agg(lambda s: s.value_counts().idxmax()))
    return st.to_dict()

def bucket_histlen(arr_like):
    """Return bucket labels for history length: 1–2, 3–5, 6–10."""
    arr = np.asarray(arr_like)
    out = np.empty(len(arr), dtype=object)
    out[arr <= 2] = "1-2"
    out[(arr >= 3) & (arr <= 5)] = "3-5"
    out[arr >= 6] = "6-10"
    return out

# ============================================================
# Trainers (with scheduler, patience, grad clipping)
# ============================================================

def _final_probs_seq(model, X, batch=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch].to(DEVICE)
            L = infer_lengths_from_padding(xb)
            out.append(torch.sigmoid(model(xb, L)).cpu())
    return torch.cat(out).numpy()

def train_seq_get_probs(model, Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=None):
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
                logits, yb, pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # validation for scheduler & early stopping
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

        _, _, auc, _ = evaluate_all(y_true, p_va)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best, pat = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best is not None: model.load_state_dict(best)
    # probs for calibration & eval
    p_tr = _final_probs_seq(model, Xtr)
    p_va = _final_probs_seq(model, Xva)
    return p_tr, p_va

def train_eval_gru_get_probs(Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=None):
    model = SponsorRiskGRU(input_dim=Xtr.shape[2], hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)
    return train_seq_get_probs(model, Xtr, ytr, Xva, yva, epochs, lr, batch, pos_weight)

def train_eval_tx_get_probs(Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=None):
    model = SponsorRiskTransformer(input_dim=Xtr.shape[2], d_model=64, nhead=4, num_layers=2,
                                   dim_feedforward=128, dropout=0.1).to(DEVICE)
    return train_seq_get_probs(model, Xtr, ytr, Xva, yva, epochs, lr, batch, pos_weight)

def train_eval_combined_get_probs(Xn_tr, Xc_tr, y_tr, L_tr,
                                  Xn_va, Xc_va, y_va, L_va, vocab_sizes,
                                  epochs=30, lr=1e-3, batch=256, pos_weight=None):
    model = CombinedGRU(num_dim=Xn_tr.shape[2], cat_vocab_sizes=vocab_sizes,
                        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
    tr_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, y_tr)
    va_ds = TensorDataset(Xn_va, Xc_va, L_va, y_va)
    tr_ld = DataLoader(tr_ds, batch_size=batch, sampler=RandomSampler(tr_ds), pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=batch, sampler=SequentialSampler(va_ds), pin_memory=True)

    best_auc, best, pat, PATIENCE = -1.0, None, 0, 5
    for ep in range(1, epochs+1):
        model.train()
        for xn, xc, l, yb in tr_ld:
            xn, xc, l, yb = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yb.to(DEVICE)
            logits = model(xn, xc, l)
            loss = F.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval(); allp, ally = [], []
        with torch.no_grad():
            for xn, xc, l, yv in va_ld:
                xn, xc, l, yv = xn.to(DEVICE), xc.to(DEVICE), l.to(DEVICE), yv.to(DEVICE)
                allp.append(torch.sigmoid(model(xn, xc, l)).cpu()); ally.append(yv.cpu())
        y_true = torch.cat(ally).numpy(); p_va = torch.cat(allp).numpy()
        eps = 1e-7
        val_logloss = -np.mean(y_true*np.log(p_va+eps) + (1-y_true)*np.log(1-p_va+eps))
        sched.step(val_logloss)

        _, _, auc, _ = evaluate_all(y_true, p_va)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best, pat = auc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best is not None: model.load_state_dict(best)

    def final_probs(Xn, Xc, L, batch=512):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch):
                xn, xc, l = Xn[i:i+batch].to(DEVICE), Xc[i:i+batch].to(DEVICE), L[i:i+batch].to(DEVICE)
                out.append(torch.sigmoid(model(xn, xc, l)).cpu())
        return torch.cat(out).numpy()

    p_tr = final_probs(Xn_tr, Xc_tr, L_tr)
    p_va = final_probs(Xn_va, Xc_va, L_va)
    return p_tr, p_va

# ============================================================
# Run everything
# ============================================================

if __name__ == "__main__":
    metrics = []
    metrics_cal = []
    slices_st = []
    slices_len = []

    # frozen split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    # sponsor_type mapping (from raw DF)
    df_all = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    st_map = sponsor_type_map(df_all)

    # ---------------- Baseline-3F + trends ----------------
    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    X3, y3, groups3 = build_baseline_samples(df, max_hist=10, use_categoricals=False, with_trends=True, verbose=False)
    X3tr, X3va = X3.iloc[tr_idx], X3.iloc[va_idx]
    y3tr, y3va = y3[tr_idx],    y3[va_idx]
    lr3 = LogisticRegression(max_iter=500, class_weight="balanced")
    lr3.fit(X3tr, y3tr)
    p3_tr = lr3.predict_proba(X3tr)[:, 1]
    p3_va = lr3.predict_proba(X3va)[:, 1]
    acc, f1, auc, pr = evaluate_all(y3va, p3_va)
    metrics.append({"model":"Baseline-3F+trends","acc":acc,"f1":f1,"auc":auc,"prauc":pr})

    p3_va_iso = calibrate_isotonic(y3tr, p3_tr, p3_va)
    acc, f1, auc, pr = evaluate_all(y3va, p3_va_iso)
    t_star, f1_star = best_f1_threshold(y3va, p3_va_iso)
    acc_thr, f1_thr, _, _ = evaluate_all(y3va, (p3_va_iso >= t_star).astype(float))
    metrics_cal.append({"model":"Baseline-3F+trends (iso)","acc":acc,"f1":f1,"auc":auc,"prauc":pr,
                        "thr":t_star,"acc_thr":acc_thr,"f1_thr":f1_thr})

    sponsors3 = groups3
    st_va = np.array([st_map.get(s, "unknown") for s in sponsors3[va_idx]])
    histlen_va = bucket_histlen(X3va["hist_len"].values)
    for st in np.unique(st_va):
        m = st_va == st
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(y3va[m], p3_va_iso[m])
            slices_st.append({"model":"Baseline-3F+trends","sponsor_type":st,"n":int(m.sum()),
                              "auc":auc_s,"prauc":pr_s})
    for b in ["1-2","3-5","6-10"]:
        m = histlen_va == b
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(y3va[m], p3_va_iso[m])
            slices_len.append({"model":"Baseline-3F+trends","bucket":b,"n":int(m.sum()),
                               "auc":auc_s,"prauc":pr_s})

    # ---------------- Baseline-7F + trends ----------------
    X7, y7, groups7 = build_baseline_samples(df, max_hist=10, use_categoricals=True, with_trends=True, verbose=False)
    X7tr, X7va = X7.iloc[tr_idx], X7.iloc[va_idx]
    y7tr, y7va = y7[tr_idx],    y7[va_idx]
    lr7 = LogisticRegression(max_iter=500, class_weight="balanced")
    lr7.fit(X7tr, y7tr)
    p7_tr = lr7.predict_proba(X7tr)[:, 1]
    p7_va = lr7.predict_proba(X7va)[:, 1]
    acc, f1, auc, pr = evaluate_all(y7va, p7_va)
    metrics.append({"model":"Baseline-7F+trends","acc":acc,"f1":f1,"auc":auc,"prauc":pr})

    p7_va_iso = calibrate_isotonic(y7tr, p7_tr, p7_va)
    acc, f1, auc, pr = evaluate_all(y7va, p7_va_iso)
    t_star, f1_star = best_f1_threshold(y7va, p7_va_iso)
    acc_thr, f1_thr, _, _ = evaluate_all(y7va, (p7_va_iso >= t_star).astype(float))
    metrics_cal.append({"model":"Baseline-7F+trends (iso)","acc":acc,"f1":f1,"auc":auc,"prauc":pr,
                        "thr":t_star,"acc_thr":acc_thr,"f1_thr":f1_thr})

    sponsors7 = groups7
    st_va = np.array([st_map.get(s, "unknown") for s in sponsors7[va_idx]])
    histlen_va = bucket_histlen(X7va["hist_len"].values)
    for st in np.unique(st_va):
        m = st_va == st
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(y7va[m], p7_va_iso[m])
            slices_st.append({"model":"Baseline-7F+trends","sponsor_type":st,"n":int(m.sum()),
                              "auc":auc_s,"prauc":pr_s})
    for b in ["1-2","3-5","6-10"]:
        m = histlen_va == b
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(y7va[m], p7_va_iso[m])
            slices_len.append({"model":"Baseline-7F+trends","bucket":b,"n":int(m.sum()),
                               "auc":auc_s,"prauc":pr_s})

    # ---------------- GRU (9ch) ----------------
    X, y, L, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
    Xtr, Xva = X[tr_idx], X[va_idx]; ytr, yva = y[tr_idx], y[va_idx]
    pos_w = torch.tensor((ytr==0).sum().item()/max((ytr==1).sum().item(),1), dtype=torch.float32) * 1.2
    p_tr, p_va = train_eval_gru_get_probs(Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=pos_w)
    acc, f1, auc, pr = evaluate_all(yva.numpy(), p_va)
    metrics.append({"model":"GRU-9ch","acc":acc,"f1":f1,"auc":auc,"prauc":pr})

    p_va_iso = calibrate_isotonic(ytr.numpy(), p_tr, p_va)
    acc, f1, auc, pr = evaluate_all(yva.numpy(), p_va_iso)
    t_star, f1_star = best_f1_threshold(yva.numpy(), p_va_iso)
    acc_thr, f1_thr, _, _ = evaluate_all(yva.numpy(), (p_va_iso >= t_star).astype(float))
    metrics_cal.append({"model":"GRU-9ch (iso)","acc":acc,"f1":f1,"auc":auc,"prauc":pr,
                        "thr":t_star,"acc_thr":acc_thr,"f1_thr":f1_thr})

    st_va = np.array([st_map.get(s, "unknown") for s in np.array(sponsors)[va_idx]])
    len_va = bucket_histlen(L[va_idx].numpy())
    for st in np.unique(st_va):
        m = st_va == st
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yva.numpy()[m], p_va_iso[m])
            slices_st.append({"model":"GRU-9ch","sponsor_type":st,"n":int(m.sum()),
                              "auc":auc_s,"prauc":pr_s})
    for b in ["1-2","3-5","6-10"]:
        m = len_va == b
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yva.numpy()[m], p_va_iso[m])
            slices_len.append({"model":"GRU-9ch","bucket":b,"n":int(m.sum()),
                               "auc":auc_s,"prauc":pr_s})

    # ---------------- Transformer (9ch) ----------------
    p_tr, p_va = train_eval_tx_get_probs(Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=256, pos_weight=pos_w)
    acc, f1, auc, pr = evaluate_all(yva.numpy(), p_va)
    metrics.append({"model":"Transformer-9ch","acc":acc,"f1":f1,"auc":auc,"prauc":pr})

    p_va_iso = calibrate_isotonic(ytr.numpy(), p_tr, p_va)
    acc, f1, auc, pr = evaluate_all(yva.numpy(), p_va_iso)
    t_star, f1_star = best_f1_threshold(yva.numpy(), p_va_iso)
    acc_thr, f1_thr, _, _ = evaluate_all(yva.numpy(), (p_va_iso >= t_star).astype(float))
    metrics_cal.append({"model":"Transformer-9ch (iso)","acc":acc,"f1":f1,"auc":auc,"prauc":pr,
                        "thr":t_star,"acc_thr":acc_thr,"f1_thr":f1_thr})

    for st in np.unique(st_va):
        m = st_va == st
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yva.numpy()[m], p_va_iso[m])
            slices_st.append({"model":"Transformer-9ch","sponsor_type":st,"n":int(m.sum()),
                              "auc":auc_s,"prauc":pr_s})
    for b in ["1-2","3-5","6-10"]:
        m = len_va == b
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yva.numpy()[m], p_va_iso[m])
            slices_len.append({"model":"Transformer-9ch","bucket":b,"n":int(m.sum()),
                               "auc":auc_s,"prauc":pr_s})

    # ---------------- Combined (9+4) ----------------
    Xn, Xc, yC, Lc, sponsorsC, vocab_sizes = build_sequences_with_cats_trends("data/aact_extracted.csv", max_seq_len=10, verbose=False)
    Xn_tr, Xn_va = Xn[tr_idx], Xn[va_idx]
    Xc_tr, Xc_va = Xc[tr_idx], Xc[va_idx]
    yC_tr, yC_va = yC[tr_idx], yC[va_idx]
    L_tr, L_va   = Lc[tr_idx], Lc[va_idx]
    pos_w2 = torch.tensor((yC_tr==0).sum().item()/max((yC_tr==1).sum().item(),1), dtype=torch.float32) * 1.2

    p_tr, p_va = train_eval_combined_get_probs(Xn_tr, Xc_tr, yC_tr, L_tr, Xn_va, Xc_va, yC_va, L_va,
                                               vocab_sizes, epochs=30, lr=1e-3, batch=256, pos_weight=pos_w2)
    acc, f1, auc, pr = evaluate_all(yC_va.numpy(), p_va)
    metrics.append({"model":"Combined-9+4","acc":acc,"f1":f1,"auc":auc,"prauc":pr})

    p_va_iso = calibrate_isotonic(yC_tr.numpy(), p_tr, p_va)
    acc, f1, auc, pr = evaluate_all(yC_va.numpy(), p_va_iso)
    t_star, f1_star = best_f1_threshold(yC_va.numpy(), p_va_iso)
    acc_thr, f1_thr, _, _ = evaluate_all(yC_va.numpy(), (p_va_iso >= t_star).astype(float))
    metrics_cal.append({"model":"Combined-9+4 (iso)","acc":acc,"f1":f1,"auc":auc,"prauc":pr,
                        "thr":t_star,"acc_thr":acc_thr,"f1_thr":f1_thr})

    st_vaC = np.array([st_map.get(s, "unknown") for s in np.array(sponsorsC)[va_idx]])
    len_vaC = bucket_histlen(Lc[va_idx].numpy())
    for st in np.unique(st_vaC):
        m = st_vaC == st
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yC_va.numpy()[m], p_va_iso[m])
            slices_st.append({"model":"Combined-9+4","sponsor_type":st,"n":int(m.sum()),
                              "auc":auc_s,"prauc":pr_s})
    for b in ["1-2","3-5","6-10"]:
        m = len_vaC == b
        if m.sum() >= 50:
            _, _, auc_s, pr_s = evaluate_all(yC_va.numpy()[m], p_va_iso[m])
            slices_len.append({"model":"Combined-9+4","bucket":b,"n":int(m.sum()),
                               "auc":auc_s,"prauc":pr_s})

    # ---------------- Save everything ----------------
    df_overall = pd.DataFrame(metrics).sort_values("auc", ascending=False)
    df_overall_cal = pd.DataFrame(metrics_cal).sort_values("auc", ascending=False)
    df_st = pd.DataFrame(slices_st).sort_values(["model","sponsor_type"])
    df_len = pd.DataFrame(slices_len).sort_values(["model","bucket"])

    print("\n=== OVERALL (uncalibrated) ===")
    print(df_overall.to_string(index=False))
    print("\n=== OVERALL (isotonic-calibrated, +best-F1 threshold) ===")
    print(df_overall_cal.to_string(index=False))

    df_overall.to_csv(os.path.join(OUTDIR, "metrics_overall.csv"), index=False)
    df_overall_cal.to_csv(os.path.join(OUTDIR, "metrics_overall_calibrated.csv"), index=False)
    df_st.to_csv(os.path.join(OUTDIR, "slices_sponsor_type.csv"), index=False)
    df_len.to_csv(os.path.join(OUTDIR, "slices_histlen.csv"), index=False)
    with open(os.path.join(OUTDIR, "metrics_overall.json"), "w") as f:
        json.dump(df_overall.to_dict(orient="records"), f, indent=2)
    with open(os.path.join(OUTDIR, "metrics_overall_calibrated.json"), "w") as f:
        json.dump(df_overall_cal.to_dict(orient="records"), f, indent=2)

    print("✅ Saved:",
          "results/metrics_overall.csv, results/metrics_overall_calibrated.csv,",
          "results/slices_sponsor_type.csv, results/slices_histlen.csv")
