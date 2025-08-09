import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

# ----------------------------
# Constants & helpers
# ----------------------------

FAIL = {"terminated", "withdrawn", "suspended"}
SUCCESS = {"completed"}

# Core phases; we’ll also try an expanded set and substring matching.
PHASE_PATTERNS_CORE = {
    "phase 1", "phase 1/2", "phase 2", "phase 2/3", "phase 3",
}
PHASE_PATTERNS_EXPAND = PHASE_PATTERNS_CORE | {"early phase 1", "phase 4"}

def _norm_lower(x) -> str:
    if pd.isna(x):
        return "unknown"
    return str(x).strip().lower()

def _norm_str_underscore(x) -> str:
    if pd.isna(x):
        return "unknown"
    return re.sub(r"\s+", "_", str(x).strip().lower())

def _first_from_pg_array(val) -> str:
    """
    Handles NULL, plain strings, and Postgres array-like strings "{a,b}".
    Returns the first token normalized with underscores.
    """
    if pd.isna(val):
        return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    return _norm_str_underscore(toks[0]) if toks else "unknown"

def _month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return 0.0
    dt = later - earlier
    return dt.days / 30.4375  # approx months

def _robust_cohort_filter(df_raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Robust cohort filter that:
      - normalizes study_type/phase/overall_status to lowercase
      - applies interventional filter (exact, then substring; skips if tiny)
      - applies phase filter (core, then expanded; substring fallback; skips if tiny)
    We DO NOT filter by status here.
    """
    df = df_raw.copy()

    for col in ["study_type", "phase", "overall_status"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_lower)
        else:
            df[col] = "unknown"

    # Interventional filter
    df_interv = df[df["study_type"] == "interventional"]
    if len(df_interv) == 0:
        df_interv = df[df["study_type"].str.contains("interventional", na=False)]
        if verbose:
            print(f"[Interventional substring] trials={len(df_interv)}")
    else:
        if verbose:
            print(f"[Interventional exact] trials={len(df_interv)}")

    # If applying interventional kills the cohort, skip it
    if len(df_interv) < 1000:
        if verbose:
            print("[Interventional] too small; skipping study_type filter.")
        df_interv = df

    # Phase filter
    def filter_by_phase(df_in: pd.DataFrame, phases: set) -> pd.DataFrame:
        # Accept values like "phase2" or "phase 2" or "phase 1/phase 2"
        canon = df_in["phase"].copy()
        # insert spaces where missing (phase2 -> phase 2) for matching
        canon = canon.str.replace(r"phase(\d)", r"phase \1", regex=True)
        canon = canon.str.replace(r"early_phase(\d)", r"early phase \1", regex=True)
        mask = canon.isin(phases)
        if mask.sum() == 0:
            pat = "|".join([re.escape(p) for p in phases])
            mask = canon.str.contains(pat, na=False)
        return df_in[mask]

    df_ph = filter_by_phase(df_interv, PHASE_PATTERNS_CORE)
    if verbose:
        print(f"[Phase core] trials={len(df_ph)}")
    if len(df_ph) < 1000:
        df_ph = filter_by_phase(df_interv, PHASE_PATTERNS_EXPAND)
        if verbose:
            print(f"[Phase expanded] trials={len(df_ph)}")
    if len(df_ph) < 1000:
        if verbose:
            print("[Phase] too small; skipping phase filter.")
        df_ph = df_interv

    # Drop rows without sponsor or start_date
    df_ph = df_ph.dropna(subset=["sponsor_name", "start_date"]).copy()
    return df_ph

# ----------------------------
# 3F builder (numeric only)
# ----------------------------

def build_sequences_rich(
    csv_path: str = "data/aact_extracted.csv",
    max_seq_len: int = 10,
    verbose: bool = False,
):
    """
    Returns:
        X: (N, T, 3) float32   -> [phase_enc, enroll_z, gap_months]
        y: (N,) float32        -> 1 fail, 0 success  (only when NEXT trial status is in {FAIL,SUCCESS})
        lengths: (N,) int64    -> true (unpadded) length
        sponsors: List[str]    -> sponsor names aligned with X/y rows
    """
    df_raw = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = _robust_cohort_filter(df_raw, verbose=verbose)

    # Sort for time order
    df = df.sort_values(["sponsor_name", "start_date"]).copy()

    # Phase encoding (encode normalized 'phase' tokens—robust)
    df["phase_raw"] = df["phase"].fillna("n/a")
    phase_encoder = LabelEncoder().fit(df["phase_raw"])
    df["phase_enc"] = phase_encoder.transform(df["phase_raw"])

    # Enrollment -> log1p + z-score
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    sequences: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    sponsors: List[str] = []
    lengths: List[int] = []

    # Build sequences per sponsor
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        # time gaps (months) between consecutive starts
        gaps = [0.0]
        for i in range(1, len(g)):
            gaps.append(_month_diff(g.loc[i, "start_date"], g.loc[i - 1, "start_date"]))
        g["gap_months"] = pd.Series(gaps).clip(lower=0.0, upper=120.0)

        for i in range(1, len(g)):
            nxt_status = _norm_lower(g.loc[i, "overall_status"])
            # label only when NEXT status is one we train on; else skip
            if (nxt_status not in FAIL) and (nxt_status not in SUCCESS):
                continue
            target = 1.0 if nxt_status in FAIL else 0.0

            hist = g.iloc[max(0, i - max_seq_len): i]

            Xn = torch.tensor(
                hist[["phase_enc", "enroll_z", "gap_months"]].values,
                dtype=torch.float32,
            )

            sequences.append(Xn)
            labels.append(torch.tensor(target, dtype=torch.float32))
            sponsors.append(sponsor)
            lengths.append(Xn.shape[0])

    if not sequences:
        return (
            torch.zeros((0, max_seq_len, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            sponsors,
        )

    X = pad_sequence(sequences, batch_first=True)  # (N, T, 3)
    y = torch.stack(labels)
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    return X, y, lengths_t, sponsors

# ----------------------------
# 7F builder (numeric + categorical indices)
# ----------------------------

def build_sequences_with_cats(
    csv_path: str = "data/aact_extracted.csv",
    max_seq_len: int = 10,
    verbose: bool = False,
):
    """
    Returns:
        X_num: (N, T, 3) float32 -> [phase_enc, enroll_z, gap_months]
        X_cat: (N, T, 4) long    -> indices for [allocation, masking, primary_purpose, intv_type]
        y: (N,) float32
        lengths: (N,) int64
        sponsors: List[str]
        vocab_sizes: Dict[str, int] -> vocab sizes for each categorical field
    """
    df_raw = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = _robust_cohort_filter(df_raw, verbose=verbose)

    # Sort for time order
    df = df.sort_values(["sponsor_name", "start_date"]).copy()

    # Numeric features (same as 3F)
    df["phase_raw"] = df["phase"].fillna("n/a")
    pe = LabelEncoder()
    df["phase_enc"] = pe.fit_transform(df["phase_raw"])

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # Categorical normalization
    df["allocation_norm"] = df.get("allocation", "unknown").apply(_norm_str_underscore)
    df["masking_norm"] = df.get("masking", "unknown").apply(_norm_str_underscore)
    df["primary_purpose_norm"] = df.get("primary_purpose", "unknown").apply(_norm_str_underscore)
    df["intv_type_norm"] = df.get("intervention_types", "unknown").apply(_first_from_pg_array)

    # Build vocabularies
    def build_vocab(series: pd.Series) -> Dict[str, int]:
        vals = series.fillna("unknown").astype(str).map(_norm_str_underscore).unique().tolist()
        vals = sorted(set(vals))
        return {v: i for i, v in enumerate(vals)}

    vocab = {
        "allocation": build_vocab(df["allocation_norm"]),
        "masking": build_vocab(df["masking_norm"]),
        "primary_purpose": build_vocab(df["primary_purpose_norm"]),
        "intv_type": build_vocab(df["intv_type_norm"]),
    }

    def map_idx(series: pd.Series, key: str) -> pd.Series:
        table = vocab[key]
        return series.fillna("unknown").map(lambda v: table.get(_norm_str_underscore(v), 0)).astype(int)

    df["alloc_idx"] = map_idx(df["allocation_norm"], "allocation")
    df["mask_idx"]  = map_idx(df["masking_norm"], "masking")
    df["purp_idx"]  = map_idx(df["primary_purpose_norm"], "primary_purpose")
    df["intv_idx"]  = map_idx(df["intv_type_norm"], "intv_type")

    X_num_list: List[torch.Tensor] = []
    X_cat_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    lengths: List[int] = []
    sponsors: List[str] = []

    for sponsor, g in df.groupby("sponsor_name"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        # time gaps (months)
        gaps = [0.0]
        for i in range(1, len(g)):
            gaps.append(_month_diff(g.loc[i, "start_date"], g.loc[i - 1, "start_date"]))
        g["gap_months"] = pd.Series(gaps).clip(lower=0.0, upper=120.0)

        for i in range(1, len(g)):
            nxt_status = _norm_lower(g.loc[i, "overall_status"])
            if (nxt_status not in FAIL) and (nxt_status not in SUCCESS):
                continue
            target = 1.0 if nxt_status in FAIL else 0.0

            hist = g.iloc[max(0, i - max_seq_len): i]

            Xn = torch.tensor(
                hist[["phase_enc", "enroll_z", "gap_months"]].values,
                dtype=torch.float32,
            )
            Xc = torch.tensor(
                hist[["alloc_idx", "mask_idx", "purp_idx", "intv_idx"]].values,
                dtype=torch.long,
            )

            X_num_list.append(Xn)
            X_cat_list.append(Xc)
            y_list.append(torch.tensor(target, dtype=torch.float32))
            lengths.append(Xn.shape[0])
            sponsors.append(sponsor)

    if not X_num_list:
        return (
            torch.zeros((0, max_seq_len, 3), dtype=torch.float32),
            torch.zeros((0, max_seq_len, 4), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            sponsors,
            {k: len(v) for k, v in vocab.items()},
        )

    # pad numeric
    X_num = pad_sequence(X_num_list, batch_first=True)  # (N, T, 3)

    # pad categorical (pad with 0 indices, which exist in vocab)
    max_T = max(xc.size(0) for xc in X_cat_list)
    def pad_cat(xc: torch.Tensor, T: int) -> torch.Tensor:
        if xc.size(0) == T:
            return xc
        pad_rows = T - xc.size(0)
        pad = torch.zeros((pad_rows, xc.size(1)), dtype=xc.dtype)
        return torch.cat([xc, pad], dim=0)

    X_cat = torch.stack([pad_cat(xc, max_T) for xc in X_cat_list], dim=0)  # (N, T, 4)

    y = torch.stack(y_list)
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    vocab_sizes = {k: len(v) for k, v in vocab.items()}

    return X_num, X_cat, y, lengths_t, sponsors, vocab_sizes

def _compute_trend_feats(hist: pd.DataFrame) -> dict:
    """Compute trend/streak features on the given history slice (<= current step)."""
    h3 = hist.tail(3)
    out = {}

    # fail rate last3
    out["prior_fail_rate_last3"] = h3["overall_status"].isin(FAIL).mean() if len(h3) else 0.0

    # enroll delta & slope last3 (uses enroll_z)
    if len(h3) >= 1:
        out["enroll_delta_last3"] = h3["enroll_z"].iloc[-1] - h3["enroll_z"].mean()
    else:
        out["enroll_delta_last3"] = 0.0
    if len(h3) >= 2:
        xs = np.arange(len(h3))
        coeffs = np.polyfit(xs, h3["enroll_z"].values, 1)
        out["enroll_slope_last3"] = float(coeffs[0])
    else:
        out["enroll_slope_last3"] = 0.0

    # phase progression last3
    out["phase_prog_last3"] = (h3["phase_enc"].iloc[-1] - h3["phase_enc"].mean()) if len(h3) else 0.0

    # gap mean last3
    out["gap_mean_last3"] = h3["gap_months"].mean() if len(h3) else 0.0

    # intervention diversity last5
    h5 = hist.tail(5)
    out["intv_diversity_last5"] = h5["intv_type_norm"].nunique() if "intv_type_norm" in hist.columns and len(h5) else 0.0
    return out


def _prep_common_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Phase encode, enrollment z, gaps; normalize cats for diversity calc."""
    df = df.copy()
    df["phase_raw"] = df["phase"].fillna("n/a")
    pe = LabelEncoder().fit(df["phase_raw"])
    df["phase_enc"] = pe.transform(df["phase_raw"])

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # cats for diversity
    df["intv_type_norm"] = df.get("intervention_types", "unknown").apply(_first_from_pg_array)

    # gaps (months) by sponsor
    df = df.sort_values(["sponsor_name","start_date"]).copy()
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


def build_sequences_rich_trends(csv_path: str = "data/aact_extracted.csv", max_seq_len: int = 10, verbose: bool=False):
    """
    Returns:
        X: (N, T, 9) float32 -> [phase_enc, enroll_z, gap_months,
                                 prior_fail_rate_last3, enroll_delta_last3, enroll_slope_last3,
                                 phase_prog_last3, gap_mean_last3, intv_diversity_last5]
        y, lengths, sponsors as before.
    """
    df_raw = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = _robust_cohort_filter(df_raw, verbose=verbose)
    df = _prep_common_numeric(df)

    seqs, labels, sponsors, lengths = [], [], [], []

    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            nxt_status = _norm_lower(g.loc[i, "overall_status"])
            if (nxt_status not in FAIL) and (nxt_status not in SUCCESS):
                continue

            hist = g.iloc[max(0, i - max_seq_len): i].copy()

            # build per-timestep rows with trend features computed at each t
            rows = []
            for t in range(len(hist)):
                h_t = hist.iloc[:t+1]  # history up to time t
                trends = _compute_trend_feats(h_t)
                rows.append([
                    float(hist["phase_enc"].iloc[t]),
                    float(hist["enroll_z"].iloc[t]),
                    float(hist["gap_months"].iloc[t]),
                    float(trends["prior_fail_rate_last3"]),
                    float(trends["enroll_delta_last3"]),
                    float(trends["enroll_slope_last3"]),
                    float(trends["phase_prog_last3"]),
                    float(trends["gap_mean_last3"]),
                    float(trends["intv_diversity_last5"]),
                ])
            Xn = torch.tensor(rows, dtype=torch.float32)

            seqs.append(Xn)
            labels.append(torch.tensor(1.0 if nxt_status in FAIL else 0.0, dtype=torch.float32))
            sponsors.append(sponsor)
            lengths.append(Xn.shape[0])

    if not seqs:
        return (
            torch.zeros((0, max_seq_len, 9), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            sponsors,
        )

    X = pad_sequence(seqs, batch_first=True)
    y = torch.stack(labels)
    L = torch.tensor(lengths, dtype=torch.int64)
    return X, y, L, sponsors


def build_sequences_with_cats_trends(csv_path: str = "data/aact_extracted.csv", max_seq_len: int = 10, verbose: bool=False):
    """
    Returns:
        X_num: (N, T, 9) float32   (same 9 numeric channels as above)
        X_cat: (N, T, 4) long      (alloc_idx, mask_idx, purp_idx, intv_idx)
        y:     (N,) float32
        lengths: (N,) int64
        sponsors: List[str]
        vocab_sizes: Dict[str, int]
        vocab_maps:  Dict[str, Dict[str, int]]   # string→index maps for each categorical
    """
    df_raw = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = _robust_cohort_filter(df_raw, verbose=verbose)
    df = _prep_common_numeric(df)

    # categorical vocabularies
    df["allocation_norm"]      = df.get("allocation", "unknown").apply(_norm_str_underscore)
    df["masking_norm"]         = df.get("masking", "unknown").apply(_norm_str_underscore)
    df["primary_purpose_norm"] = df.get("primary_purpose", "unknown").apply(_norm_str_underscore)
    df["intv_type_norm"]       = df["intv_type_norm"]  # already normalized above

    def build_vocab(series: pd.Series) -> Dict[str, int]:
        vals = series.fillna("unknown").astype(str).map(_norm_str_underscore).unique().tolist()
        return {v: i for i, v in enumerate(sorted(set(vals)))}

    vocab = {
        "allocation": build_vocab(df["allocation_norm"]),
        "masking": build_vocab(df["masking_norm"]),
        "primary_purpose": build_vocab(df["primary_purpose_norm"]),
        "intv_type": build_vocab(df["intv_type_norm"]),
    }

    def map_idx(series: pd.Series, key: str) -> pd.Series:
        table = vocab[key]
        return series.fillna("unknown").map(lambda v: table.get(_norm_str_underscore(v), 0)).astype(int)

    df["alloc_idx"] = map_idx(df["allocation_norm"], "allocation")
    df["mask_idx"]  = map_idx(df["masking_norm"], "masking")
    df["purp_idx"]  = map_idx(df["primary_purpose_norm"], "primary_purpose")
    df["intv_idx"]  = map_idx(df["intv_type_norm"], "intv_type")

    Xn_list, Xc_list, y_list, L_list, sponsors = [], [], [], [], []

    for sponsor, g in df.groupby("sponsor_name"):
        g = g.sort_values("start_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(1, len(g)):
            nxt_status = _norm_lower(g.loc[i, "overall_status"])
            if (nxt_status not in FAIL) and (nxt_status not in SUCCESS):
                continue

            hist = g.iloc[max(0, i - max_seq_len): i].copy()

            rows_num, rows_cat = [], []
            for t in range(len(hist)):
                h_t = hist.iloc[:t+1]
                trends = _compute_trend_feats(h_t)
                rows_num.append([
                    float(hist["phase_enc"].iloc[t]),
                    float(hist["enroll_z"].iloc[t]),
                    float(hist["gap_months"].iloc[t]),
                    float(trends["prior_fail_rate_last3"]),
                    float(trends["enroll_delta_last3"]),
                    float(trends["enroll_slope_last3"]),
                    float(trends["phase_prog_last3"]),
                    float(trends["gap_mean_last3"]),
                    float(trends["intv_diversity_last5"]),
                ])
                rows_cat.append([
                    int(hist["alloc_idx"].iloc[t]),
                    int(hist["mask_idx"].iloc[t]),
                    int(hist["purp_idx"].iloc[t]),
                    int(hist["intv_idx"].iloc[t]),
                ])

            Xn = torch.tensor(rows_num, dtype=torch.float32)
            Xc = torch.tensor(rows_cat, dtype=torch.long)

            Xn_list.append(Xn)
            Xc_list.append(Xc)
            y_list.append(torch.tensor(1.0 if nxt_status in FAIL else 0.0, dtype=torch.float32))
            L_list.append(Xn.shape[0])
            sponsors.append(sponsor)

    if not Xn_list:
        return (
            torch.zeros((0, max_seq_len, 9), dtype=torch.float32),
            torch.zeros((0, max_seq_len, 4), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            sponsors,
            {k: len(v) for k, v in vocab.items()},
            vocab,  # return maps too (even if empty)
        )
    X_num = pad_sequence(Xn_list, batch_first=True)
    max_T = max(xc.size(0) for xc in Xc_list)
    def pad_cat(xc: torch.Tensor, T: int) -> torch.Tensor:
        if xc.size(0) == T: return xc
        pad_rows = T - xc.size(0)
        return torch.cat([xc, torch.zeros((pad_rows, xc.size(1)), dtype=xc.dtype)], dim=0)
    X_cat = torch.stack([pad_cat(xc, max_T) for xc in Xc_list], dim=0)

    y = torch.stack(y_list)
    L = torch.tensor(L_list, dtype=torch.int64)
    vocab_sizes = {k: len(v) for k, v in vocab.items()}
    vocab_maps = vocab  # clearer name at return site
    return X_num, X_cat, y, L, sponsors, vocab_sizes, vocab_maps

