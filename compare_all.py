# compare_all.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.prepare_sequences import (
    build_sequences_rich,        # 3F numeric: phase_enc, enroll_z, gap_months
    build_sequences_with_cats,   # 7F = 3F + 4 categoricals (indices + vocab)
)

from src.model import SponsorRiskGRU
from src.model_transformer import SponsorRiskTransformer
from src.model_combined import CombinedGRU
from src.train import fit  # for GRU/Transformer

# -------------------------
# Helpers
# -------------------------
FAIL = {"terminated", "withdrawn", "suspended"}
SUCCESS = {"completed"}

def _norm_status(s):
    if pd.isna(s): return "unknown"
    return str(s).strip().lower()

def _first_from_pg_array(val):
    if pd.isna(val): return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    return (toks[0].strip().lower() if toks else "unknown")

def build_baseline_samples(df, max_hist=10, use_categoricals=False):
    """Build per-(sponsor,next_trial) samples using aggregated features."""
    df = df.dropna(subset=["sponsor_name", "start_date"]).copy()
    df["status_norm"] = df["overall_status"].apply(_norm_status)
    df = df.sort_values(["sponsor_name", "start_date"])

    # numeric to match 3F
    df["phase"] = df["phase"].fillna("N/A")
    pe = LabelEncoder().fit(df["phase"])
    df["phase_enc"] = pe.transform(df["phase"])

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # time gaps per sponsor
    gaps = []
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date")
        d = [0.0]
        for i in range(1, len(g)):
            dt = (g["start_date"].iloc[i] - g["start_date"].iloc[i-1]).days / 30.4375
            d.append(max(0.0, min(120.0, dt)))
        gaps.extend(d)
    df["gap_months"] = pd.Series(gaps, index=df.index)

    # categorical normals (for 7F)
    if use_categoricals:
        df["allocation_norm"]      = df.get("allocation", "unknown").astype(str).str.strip().str.lower()
        df["masking_norm"]         = df.get("masking", "unknown").astype(str).str.strip().str.lower()
        df["primary_purpose_norm"] = df.get("primary_purpose", "unknown").astype(str).str.strip().str.lower()
        df["intv_type_norm"]       = df.get("intervention_types", "unknown").apply(_first_from_pg_array)

        cats = ["allocation_norm","masking_norm","primary_purpose_norm","intv_type_norm"]
        cat_vals = {c: sorted(df[c].fillna("unknown").astype(str).unique().tolist()) for c in cats}
        cat_idx  = {c: {v:i for i,v in enumerate(vals)} for c,vals in cat_vals.items()}

    rows, sponsors = [], []
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_hist):i]
            nxt = _norm_status(g.loc[i, "status_norm"])
            y = 1 if nxt in FAIL else (0 if nxt in SUCCESS else 0)

            feats = {
                # 3F aggregates
                "phase_enc_mean": hist["phase_enc"].mean(),
                "phase_enc_last": hist["phase_enc"].iloc[-1],
                "enroll_z_mean":  hist["enroll_z"].mean(),
                "gap_months_mean": hist["gap_months"].mean(),
                "hist_len": len(hist),
                "prior_fail_rate": hist["status_norm"].apply(_norm_status).isin(FAIL).mean() if len(hist) else 0.0,
            }

            if use_categoricals:
                # add one-hot of last categorical state
                last_alloc = hist.get("allocation_norm", pd.Series(["unknown"])).iloc[-1]
                last_mask  = hist.get("masking_norm", pd.Series(["unknown"])).iloc[-1]
                last_purp  = hist.get("primary_purpose_norm", pd.Series(["unknown"])).iloc[-1]
                last_intv  = hist.get("intv_type_norm", pd.Series(["unknown"])).iloc[-1]
                for cname, last_val in [
                    ("allocation_norm", last_alloc),
                    ("masking_norm", last_mask),
                    ("primary_purpose_norm", last_purp),
                    ("intv_type_norm", last_intv),
                ]:
                    K = len(cat_idx[cname])
                    oh = np.zeros(K, dtype=float)
                    j = cat_idx[cname].get(str(last_val), None)
                    if j is not None:
                        oh[j] = 1.0
                    for k in range(K):
                        feats[f"{cname}__{k}"] = oh[k]

            rows.append((feats, y))
            sponsors.append(sponsor)

    X = pd.DataFrame([r[0] for r in rows]).fillna(0.0)
    y = np.array([r[1] for r in rows], dtype=int)
    sponsors = np.array(sponsors)
    return X, y, sponsors

def evaluate_probs(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 0) load frozen split
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    metrics = []

    # ----------------- Baseline-3F -----------------
    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    X3, y3, groups3 = build_baseline_samples(df, max_hist=10, use_categoricals=False)
    X3tr, X3va = X3.iloc[tr_idx], X3.iloc[va_idx]
    y3tr, y3va = y3[tr_idx],    y3[va_idx]

    lr3 = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    lr3.fit(X3tr, y3tr)
    p3 = lr3.predict_proba(X3va)[:, 1]
    acc, f1, auc = evaluate_probs(y3va, p3)
    print(f"[Baseline-3F] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
    metrics.append({"model":"Baseline-3F","acc":acc,"f1":f1,"auc":auc})

    # ----------------- Baseline-7F -----------------
    X7, y7, groups7 = build_baseline_samples(df, max_hist=10, use_categoricals=True)
    X7tr, X7va = X7.iloc[tr_idx], X7.iloc[va_idx]
    y7tr, y7va = y7[tr_idx],    y7[va_idx]

    lr7 = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    lr7.fit(X7tr, y7tr)
    p7 = lr7.predict_proba(X7va)[:, 1]
    acc, f1, auc = evaluate_probs(y7va, p7)
    print(f"[Baseline-7F] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
    metrics.append({"model":"Baseline-7F","acc":acc,"f1":f1,"auc":auc})

    # ----------------- GRU-3F -----------------
    X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    model_gru = SponsorRiskGRU(input_dim=X.shape[2], hidden_dim=64, num_layers=1, dropout=0.1)
    model_gru, stats = fit(model_gru, Xtr, ytr, Xva, yva, epochs=12, lr=1e-3, batch_size=128, early_stopping=3)
    # re-eval on val to get probs (fit returns only auc)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gru.to(device).eval()
    with torch.no_grad():
        from src.train import infer_lengths_from_padding
        lengths_va = infer_lengths_from_padding(Xva.to(device))
        logits = model_gru(Xva.to(device), lengths_va)
        p = torch.sigmoid(logits).cpu().numpy()
    acc, f1, auc = evaluate_probs(yva.numpy(), p)
    print(f"[GRU-3F] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
    metrics.append({"model":"GRU-3F","acc":acc,"f1":f1,"auc":auc})

    # ----------------- Transformer-3F -----------------
    model_tx = SponsorRiskTransformer(input_dim=X.shape[2], d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
    model_tx, stats = fit(model_tx, Xtr, ytr, Xva, yva, epochs=12, lr=1e-3, batch_size=128, early_stopping=3)
    model_tx.to(device).eval()
    with torch.no_grad():
        from src.train import infer_lengths_from_padding
        lengths_va = infer_lengths_from_padding(Xva.to(device))
        logits = model_tx(Xva.to(device), lengths_va)
        p = torch.sigmoid(logits).cpu().numpy()
    acc, f1, auc = evaluate_probs(yva.numpy(), p)
    print(f"[Transformer-3F] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
    metrics.append({"model":"Transformer-3F","acc":acc,"f1":f1,"auc":auc})

    # ----------------- Combined-7F -----------------
    Xn, Xc, yC, L, sponsorsC, vocab_sizes = build_sequences_with_cats("data/aact_extracted.csv", max_seq_len=10)
    Xn_tr, Xc_tr, yC_tr, L_tr = Xn[tr_idx], Xc[tr_idx], yC[tr_idx], L[tr_idx]
    Xn_va, Xc_va, yC_va, L_va = Xn[va_idx], Xc[va_idx], yC[va_idx], L[va_idx]

    model_cmb = CombinedGRU(
        num_dim=Xn.shape[2],
        cat_vocab_sizes=vocab_sizes,
        emb_dim=16,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1
    )
    optim = torch.optim.Adam(model_cmb.parameters(), lr=1e-3)
    pos_weight = torch.tensor( ( (yC_tr==0).sum().item() / max((yC_tr==1).sum().item(), 1) ), dtype=torch.float32)

    model_cmb.to(device)
    best_auc, best_state, patience = -1.0, None, 3
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
    tr_ds = TensorDataset(Xn_tr, Xc_tr, L_tr, yC_tr)
    va_ds = TensorDataset(Xn_va, Xc_va, L_va, yC_va)
    tr_ld = DataLoader(tr_ds, batch_size=128, sampler=RandomSampler(tr_ds))
    va_ld = DataLoader(va_ds, batch_size=128, sampler=SequentialSampler(va_ds))

    for ep in range(1, 13):
        model_cmb.train(); tr_loss = 0.0
        for xn, xc, l, yt in tr_ld:
            xn, xc, l, yt = xn.to(device), xc.to(device), l.to(device), yt.to(device)
            logits = model_cmb(xn, xc, l)
            loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
            tr_loss += loss.item() * yt.size(0)

        # eval
        model_cmb.eval(); all_p, all_y, va_loss = [], [], 0.0
        with torch.no_grad():
            for xn, xc, l, yv in va_ld:
                xn, xc, l, yv = xn.to(device), xc.to(device), l.to(device), yv.to(device)
                logits = model_cmb(xn, xc, l)
                loss = F.binary_cross_entropy_with_logits(logits, yv)
                va_loss += loss.item() * yv.size(0)
                all_p.append(torch.sigmoid(logits).cpu()); all_y.append(yv.cpu())
        y_true = torch.cat(all_y).numpy()
        y_prob = torch.cat(all_p).numpy()
        acc_, f1_, auc_ = evaluate_probs(y_true, y_prob)
        print(f"[Combined-7F][Ep{ep:02d}] acc={acc_:.3f} | f1={f1_:.3f} | auc={auc_:.3f}")

        if not np.isnan(auc_) and auc_ > best_auc:
            best_auc = auc_; best_state = {k:v.cpu().clone() for k,v in model_cmb.state_dict().items()}
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping for Combined-7F.")
                break

    if best_state is not None:
        model_cmb.load_state_dict(best_state)

    # final val metrics
    model_cmb.eval()
    with torch.no_grad():
        all_p, all_y = [], []
        for xn, xc, l, yv in va_ld:
            xn, xc, l, yv = xn.to(device), xc.to(device), l.to(device), yv.to(device)
            logits = model_cmb(xn, xc, l)
            all_p.append(torch.sigmoid(logits).cpu()); all_y.append(yv.cpu())
    y_true = torch.cat(all_y).numpy()
    y_prob = torch.cat(all_p).numpy()
    acc, f1, auc = evaluate_probs(y_true, y_prob)
    print(f"[Combined-7F] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
    metrics.append({"model":"Combined-7F","acc":acc,"f1":f1,"auc":auc})

    # ----------------- Save table -----------------
    dfm = pd.DataFrame(metrics)
    print("\n=== SUMMARY ===")
    print(dfm.sort_values("auc", ascending=False).to_string(index=False))

    dfm.to_csv("results/metrics.csv", index=False)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved results to results/metrics.(csv|json)")
