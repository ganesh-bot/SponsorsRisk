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

PHASE_PATTERNS_CORE = {"phase 1","phase 1/2","phase 2","phase 2/3","phase 3"}
PHASE_PATTERNS_EXPAND = PHASE_PATTERNS_CORE | {"early phase 1","phase 4"}

def _norm_lower(x):
    if pd.isna(x): return "unknown"
    return str(x).strip().lower()

def _norm_str_underscore(x):
    if pd.isna(x): return "unknown"
    return re.sub(r"\s+", "_", str(x).strip().lower())

def _first_from_pg_array(val):
    if pd.isna(val): return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    return _norm_str_underscore(toks[0]) if toks else "unknown"

def label_from_status(status_norm: str) -> int:
    if status_norm in FAIL: return 1
    if status_norm in SUCCESS: return 0
    return 0

def _robust_interventional_and_phase(df, verbose=True):
    df = df.copy()
    for col in ["study_type", "phase", "overall_status"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_lower)
        else:
            df[col] = "unknown"

    if verbose:
        print("study_type unique (top 10):", df["study_type"].value_counts().head(10).to_dict())
        print("phase unique (top 10):", df["phase"].value_counts().head(10).to_dict())
        print(f"[Start] trials={len(df)}, sponsors={df['sponsor_name'].nunique()}")

    df_interv = df[df["study_type"] == "interventional"]
    if len(df_interv) == 0:
        df_interv = df[df["study_type"].str.contains("interventional", na=False)]
        if verbose: print(f"[Interventional substring] trials={len(df_interv)}")
    else:
        if verbose: print(f"[Interventional exact] trials={len(df_interv)}")

    if len(df_interv) < 1000:
        if verbose: print("[Interventional] too small; skipping study_type filter.")
        df_interv = df

    # Try phases; if tiny, skip
    def _canon_phase(s: pd.Series) -> pd.Series:
        s = s.str.replace(r"phase(\d)", r"phase \1", regex=True)
        s = s.str.replace(r"early_phase(\d)", r"early phase \1", regex=True)
        return s

    def _filter_phase(df_in, phases):
        canon = _canon_phase(df_in["phase"])
        mask = canon.isin(phases)
        if mask.sum() == 0:
            pat = "|".join([re.escape(p) for p in phases])
            mask = canon.str.contains(pat, na=False)
        return df_in[mask]

    df_ph = _filter_phase(df_interv, PHASE_PATTERNS_CORE)
    if verbose: print(f"[Phase core] trials={len(df_ph)}")
    if len(df_ph) < 1000:
        df_ph = _filter_phase(df_interv, PHASE_PATTERNS_EXPAND)
        if verbose: print(f"[Phase expanded] trials={len(df_ph)}")
    if len(df_ph) < 1000:
        if verbose: print("[Phase] too small; skipping phase filter.")
        df_ph = df_interv

    df_ph = df_ph.dropna(subset=["sponsor_name","start_date"]).copy()
    df_ph = df_ph.sort_values(["sponsor_name","start_date"])
    if verbose:
        s2 = (df_ph.groupby("sponsor_name").size() >= 2).sum()
        print(f"[After filters] trials={len(df_ph)}, sponsors={df_ph['sponsor_name'].nunique()}, sponsors>=2={s2}")
    return df_ph

def _compute_gaps(df):
    gaps = []
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date")
        d = [0.0]
        for i in range(1, len(g)):
            dt = (g["start_date"].iloc[i] - g["start_date"].iloc[i-1]).days / 30.4375
            d.append(max(0.0, min(120.0, dt)))
        gaps.extend(d)
    df["gap_months"] = pd.Series(gaps, index=df.index)
    return df

def build_baseline_samples(df, max_hist=10, use_categoricals=False, with_trends=True, verbose=True):
    df = _robust_interventional_and_phase(df, verbose=verbose)

    # 3F numeric prep
    df["phase_raw"] = df["phase"].fillna("n/a")
    pe = LabelEncoder().fit(df["phase_raw"])
    df["phase_enc"] = pe.transform(df["phase_raw"])

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    df = _compute_gaps(df)

    # categoricals
    if use_categoricals:
        df["allocation_norm"]      = df.get("allocation", "unknown").apply(_norm_str_underscore)
        df["masking_norm"]         = df.get("masking", "unknown").apply(_norm_str_underscore)
        df["primary_purpose_norm"] = df.get("primary_purpose", "unknown").apply(_norm_str_underscore)
        df["intv_type_norm"]       = df.get("intervention_types", "unknown").apply(_first_from_pg_array)
        cats = ["allocation_norm","masking_norm","primary_purpose_norm","intv_type_norm"]
        cat_vals = {c: sorted(df[c].fillna("unknown").astype(str).unique().tolist()) for c in cats}
        cat_idx  = {c: {v:i for i,v in enumerate(vals)} for c,vals in cat_vals.items()}

    rows, sponsors = [], []
    kept = 0
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2: 
            continue
        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_hist):i]
            nxt = _norm_lower(g.loc[i, "overall_status"])
            if (nxt not in FAIL) and (nxt not in SUCCESS):
                continue
            y = label_from_status(nxt)

            # ---- base aggregates (3F-ish) ----
            feats = {
                "phase_enc_mean": hist["phase_enc"].mean(),
                "phase_enc_last": hist["phase_enc"].iloc[-1],
                "enroll_z_mean":  hist["enroll_z"].mean(),
                "gap_months_mean": hist["gap_months"].mean(),
                "hist_len": len(hist),
                "prior_fail_rate": hist["overall_status"].isin(FAIL).mean() if len(hist) else 0.0,
            }

            # ---- trends / streaks (NEW) ----
            if with_trends:
                h3 = hist.tail(3)
                # fail rate last3
                feats["prior_fail_rate_last3"] = h3["overall_status"].isin(FAIL).mean() if len(h3) else 0.0

                # enroll delta & slope last3
                if len(h3) >= 1:
                    feats["enroll_delta_last3"] = h3["enroll_z"].iloc[-1] - h3["enroll_z"].mean()
                else:
                    feats["enroll_delta_last3"] = 0.0
                if len(h3) >= 2:
                    # slope via simple linear fit on index vs enroll_z
                    xs = np.arange(len(h3))
                    coeffs = np.polyfit(xs, h3["enroll_z"].values, 1)
                    feats["enroll_slope_last3"] = float(coeffs[0])
                else:
                    feats["enroll_slope_last3"] = 0.0

                # phase progression last3
                feats["phase_prog_last3"] = (h3["phase_enc"].iloc[-1] - h3["phase_enc"].mean()) if len(h3) else 0.0

                # gap mean last3
                feats["gap_mean_last3"] = h3["gap_months"].mean() if len(h3) else 0.0

                # intervention diversity last5
                h5 = hist.tail(5)
                if "intv_type_norm" in hist.columns and len(h5):
                    feats["intv_diversity_last5"] = h5["intv_type_norm"].nunique()
                else:
                    feats["intv_diversity_last5"] = 0.0

            # ---- categoricals one-hot (optional) ----
            if use_categoricals:
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
                    K = len(cat_idx[cname])
                    oh = np.zeros(K, dtype=float)
                    j = cat_idx[cname].get(str(last_val), None)
                    if j is not None:
                        oh[j] = 1.0
                    for k in range(K):
                        feats[f"{cname}__{k}"] = oh[k]

            rows.append((feats, y))
            sponsors.append(sponsor)
            kept += 1

    if kept == 0:
        raise SystemExit("No baseline samples constructed after filters.")

    X = pd.DataFrame([r[0] for r in rows]).fillna(0.0)
    y = np.array([r[1] for r in rows], dtype=int)
    sponsors = np.array(sponsors)
    if verbose:
        print(f"[Samples] N={len(X)} | Positives={y.sum()} ({y.mean():.3%}) | Features={X.shape[1]}")
    return X, y, sponsors

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["3f","7f"], default="3f",
                    help="3f: numeric-only (+ trends); 7f: numeric + categorical one-hots (+ trends)")
    ap.add_argument("--no_trends", action="store_true", help="disable trend/streak features")
    ap.add_argument("--dump_features_csv", default=None,
                    help="If set, write the feature matrix used for this run to CSV for SHAP (path like results/baseline_features.csv)")
    args = ap.parse_args()

    df = pd.read_csv("data/aact_extracted.csv", parse_dates=["start_date"])

    X, y, groups = build_baseline_samples(
        df,
        max_hist=10,
        use_categoricals=(args.variant=="7f"),
        with_trends=(not args.no_trends),
        verbose=True
    )
    print("Baseline shape:", X.shape, "| Label dist:", dict(zip(*np.unique(y, return_counts=True))))

    # --- use the frozen sponsor-safe split ---
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xva)[:, 1]
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(yva, yhat)
    f1  = f1_score(yva, yhat, zero_division=0)
    try:
        auc = roc_auc_score(yva, p)
    except ValueError:
        auc = float("nan")

    print(f"[Baseline {args.variant.upper()}]{' (no_trends)' if args.no_trends else ''} "
        f"acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

    # ---- optional: dump features for SHAP ----
    if args.dump_features_csv:
        import pathlib
        pathlib.Path("results").mkdir(parents=True, exist_ok=True)
        # dump the SAME features the model just used (train split is enough)
        Xcols = [f"X_{c}" for c in Xtr.columns]
        import pandas as pd
        out = pd.DataFrame(Xtr, columns=Xcols)
        out["y"] = ytr.astype(int)
        out.to_csv(args.dump_features_csv, index=False)
        print(f"[dump] wrote {args.dump_features_csv} with shape {out.shape}")