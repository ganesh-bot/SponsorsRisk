# run_baseline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

FAIL = {"terminated", "withdrawn", "suspended"}
SUCCESS = {"completed"}

def norm_status(s):
    if pd.isna(s): return "unknown"
    return str(s).strip().lower()

def make_samples(df, max_hist=10):
    """
    Build per-(sponsor, next_trial) samples:
      X = aggregates over last <= max_hist prior trials
      y = next_trial is failure?
    """
    rows = []
    # encode phase once
    df["phase"] = df["phase"].fillna("N/A")
    pe = LabelEncoder().fit(df["phase"])
    df["phase_enc"] = pe.transform(df["phase"])

    # clean numerics
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])

    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2: 
            continue
        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_hist):i]
            nxt = norm_status(g.loc[i, "overall_status"])
            y = 1 if nxt in FAIL else (0 if nxt in SUCCESS else 0)

            # simple aggregates
            feats = {
                "phase_enc_mean": hist["phase_enc"].mean(),
                "phase_enc_last": hist["phase_enc"].iloc[-1],
                "enroll_log1p_mean": hist["enroll_log1p"].mean(),
                "enroll_log1p_max": hist["enroll_log1p"].max(),
                "hist_len": len(hist),
                # rudimentary prior fail-rate
                "prior_fail_rate": (hist["overall_status"].apply(norm_status).isin(FAIL).mean() if len(hist) else 0.0),
            }
            rows.append((sponsor, y, feats))

    if not rows:
        raise SystemExit("No baseline rows constructed. Check input data.")

    sponsors, y, Xdicts = zip(*rows)
    X = pd.DataFrame(Xdicts).fillna(0.0)
    y = np.array(y, dtype=int)
    sponsors = np.array(sponsors)
    return X, y, sponsors

if __name__ == "__main__":
    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])
    # sanity filter
    df = df.dropna(subset=["sponsor_name", "start_date"])

    X, y, groups = make_samples(df, max_hist=10)
    print("Baseline feature shape:", X.shape)
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    # group-wise stratified CV (5-fold)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs, f1s = [], [], []

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=groups), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        # class_weight="balanced" for imbalance
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)
        clf.fit(Xtr, ytr)

        p = clf.predict_proba(Xva)[:, 1]
        yhat = (p >= 0.5).astype(int)

        acc = accuracy_score(yva, yhat)
        f1  = f1_score(yva, yhat, zero_division=0)
        try:
            auc = roc_auc_score(yva, p)
        except ValueError:
            auc = float("nan")

        aucs.append(auc); accs.append(acc); f1s.append(f1)
        print(f"[Fold {fold}] acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

    print(f"\nBaseline (LogReg, 5-fold, sponsor-grouped) "
          f"ACC={np.nanmean(accs):.3f} | F1={np.nanmean(f1s):.3f} | AUC={np.nanmean(aucs):.3f}")
