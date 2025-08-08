# run_baseline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import argparse
import re

FAIL = {"terminated", "withdrawn", "suspended"}
SUCCESS = {"completed"}

def _norm_status(s):
    if pd.isna(s): return "unknown"
    return str(s).strip().lower()

def _norm_str(x):
    if pd.isna(x): return "unknown"
    return re.sub(r"\s+", "_", str(x).strip().lower())

def _first_from_pg_array(val):
    if pd.isna(val): return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    return _norm_str(toks[0]) if toks else "unknown"

def build_baseline_samples(df, max_hist=10, use_categoricals=False):
    # sort
    df = df.dropna(subset=["sponsor_name", "start_date"]).copy()
    df["status_norm"] = df["overall_status"].apply(_norm_status)
    df = df.sort_values(["sponsor_name", "start_date"])

    # 3F numeric prep to mirror sequence builder
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

    # 4 extra categoricals (if requested): last values only
    if use_categoricals:
        df["allocation_norm"]      = df.get("allocation", "unknown").apply(_norm_str)
        df["masking_norm"]         = df.get("masking", "unknown").apply(_norm_str)
        df["primary_purpose_norm"] = df.get("primary_purpose", "unknown").apply(_norm_str)
        df["intv_type_norm"]       = df.get("intervention_types", "unknown").apply(_first_from_pg_array)

        cats = ["allocation_norm","masking_norm","primary_purpose_norm","intv_type_norm"]
        # build vocab for one-hots
        cat_vals = {c: sorted(df[c].fillna("unknown").astype(str).unique().tolist()) for c in cats}
        cat_idx  = {c: {v:i for i,v in enumerate(vals)} for c,vals in cat_vals.items()}

    rows, sponsors = [], []
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2: 
            continue
        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_hist):i]

            y = 1 if _norm_status(g.loc[i, "status_norm"]) in FAIL else (
                0 if _norm_status(g.loc[i, "status_norm"]) in SUCCESS else 0
            )

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
                # one-hot last-values for 4 cats
                last_alloc = hist["allocation_norm"].iloc[-1] if "allocation_norm" in hist else "unknown"
                last_mask  = hist["masking_norm"].iloc[-1] if "masking_norm" in hist else "unknown"
                last_purp  = hist["primary_purpose_norm"].iloc[-1] if "primary_purpose_norm" in hist else "unknown"
                last_intv  = hist["intv_type_norm"].iloc[-1] if "intv_type_norm" in hist else "unknown"

                for cname, last_val in [
                    ("allocation_norm", last_alloc),
                    ("masking_norm", last_mask),
                    ("primary_purpose_norm", last_purp),
                    ("intv_type_norm", last_intv),
                ]:
                    # create one-hot
                    K = len(cat_idx[cname])
                    oh = np.zeros(K, dtype=float)
                    j = cat_idx[cname].get(last_val, None)
                    if j is not None:
                        oh[j] = 1.0
                    # append as multiple columns
                    for k in range(K):
                        feats[f"{cname}__{k}"] = oh[k]

            rows.append((feats, y))
            sponsors.append(sponsor)

    if not rows:
        raise SystemExit("No baseline samples constructed.")

    X = pd.DataFrame([r[0] for r in rows]).fillna(0.0)
    y = np.array([r[1] for r in rows], dtype=int)
    sponsors = np.array(sponsors)
    return X, y, sponsors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["3f","7f"], default="3f",
                    help="3f: numeric-only; 7f: numeric + categorical one-hots")
    args = ap.parse_args()

    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    X, y, groups = build_baseline_samples(df, max_hist=10, use_categoricals=(args.variant=="7f"))
    print("Baseline shape:", X.shape, "| Label dist:", dict(zip(*np.unique(y, return_counts=True))))

    # load the same sponsor-safe indices
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xva)[:, 1]
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(yva, yhat)
    f1  = f1_score(yva, yhat, zero_division=0)
    try:
        auc = roc_auc_score(yva, p)
    except ValueError:
        auc = float("nan")

    print(f"[Baseline {args.variant.upper()}] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
