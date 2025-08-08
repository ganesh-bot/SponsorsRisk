# prepare_sequences.py (replace previous)
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import numpy as np

FAIL_STATUSES = {
    "terminated", "withdrawn", "suspended"  # broadened failure list
}
SUCCESS_STATUSES = {"completed"}

def normalize_status(s):
    if pd.isna(s):
        return "unknown"
    return str(s).strip().lower()

def build_sequences(
    csv_path="data/aact_extracted.csv",
    max_seq_len=10
):
    df = pd.read_csv(csv_path, parse_dates=["start_date"])

    # drop rows without sponsor or start_date
    df = df.dropna(subset=["sponsor_name", "start_date"])

    # normalize statuses
    df["status_norm"] = df["overall_status"].apply(normalize_status)

    # show some quick counts (optional)
    # print(df["status_norm"].value_counts().head(20))

    df = df.sort_values(by=["sponsor_name", "start_date"])

    # Encode phase
    df['phase'] = df['phase'].fillna("N/A")
    phase_encoder = LabelEncoder()
    df['phase_enc'] = phase_encoder.fit_transform(df['phase'])

    sequences, labels, sponsors = [], [], []

    for sponsor, group in df.groupby("sponsor_name"):
        trials = group.reset_index(drop=True)
        if len(trials) < 2:
            continue

        for i in range(1, len(trials)):
            history = trials.iloc[max(0, i - max_seq_len):i]

            # numerical safety for enrollment
            feat_list = []
            for _, row in history.iterrows():
                phase_enc = row['phase_enc']
                enroll = row['enrollment']
                if pd.isna(enroll):
                    enroll = 0.0
                feat_list.append([phase_enc, float(enroll)])

            feature_tensor = torch.tensor(feat_list, dtype=torch.float32)

            nxt = trials.loc[i, 'status_norm']
            target = 1 if nxt in FAIL_STATUSES else (0 if nxt in SUCCESS_STATUSES else 0)
            # Note: treating non-completed (and not in explicit fail list) as 0 to keep labels sane.
            # You can also drop unknowns if desired.

            sequences.append(feature_tensor)
            labels.append(torch.tensor(target, dtype=torch.float32))
            sponsors.append(sponsor)

    # pad
    X = pad_sequence(sequences, batch_first=True)  # (N, T, D)
    y = torch.stack(labels)                        # (N,)
    return X, y, sponsors


# append to prepare_sequences.py

def _norm_status(s):
    if pd.isna(s): return "unknown"
    return str(s).strip().lower()

def _month_diff(later, earlier):
    if pd.isna(later) or pd.isna(earlier): return 0.0
    dt = later - earlier
    return dt.days / 30.4375  # approx months

def build_sequences_rich(
    csv_path="data/aact_extracted.csv",
    max_seq_len=10
):
    df = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = df.dropna(subset=["sponsor_name", "start_date"])

    # normalize status
    df["status_norm"] = df["overall_status"].apply(_norm_status)

    # phase encoding
    df["phase"] = df["phase"].fillna("N/A")
    phase_encoder = LabelEncoder()
    df["phase_enc"] = phase_encoder.fit_transform(df["phase"])

    # clean enrollment -> log1p & clip (robust to huge outliers)
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    # z-score (avoid div by zero)
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # sort for time order
    df = df.sort_values(["sponsor_name", "start_date"])

    sequences, labels, sponsors, lengths = [], [], [], []

    for sponsor, g in df.groupby("sponsor_name"):
        g = g.reset_index(drop=True)
        if len(g) < 2: 
            continue

        # precompute time gaps (months) between consecutive trials
        # gap for trial t = months between start_date[t] and start_date[t-1]
        gaps = [0.0]
        for i in range(1, len(g)):
            gaps.append(_month_diff(g.loc[i, "start_date"], g.loc[i-1, "start_date"]))
        g["gap_months"] = gaps
        # clip extreme gaps to keep scale reasonable
        g["gap_months"] = g["gap_months"].clip(lower=0.0, upper=120.0)  # cap at 10 years

        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_seq_len):i]

            feat = torch.tensor([
                [
                    float(r["phase_enc"]),
                    float(r["enroll_z"]),
                    float(r["gap_months"]),
                ]
                for _, r in hist.iterrows()
            ], dtype=torch.float32)

            nxt = _norm_status(g.loc[i, "status_norm"])
            target = 1.0 if nxt in FAIL_STATUSES else (0.0 if nxt in SUCCESS_STATUSES else 0.0)

            sequences.append(feat)
            labels.append(torch.tensor(target, dtype=torch.float32))
            sponsors.append(sponsor)
            lengths.append(feat.shape[0])

    X = pad_sequence(sequences, batch_first=True)  # (N, T, D=3)
    y = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return X, y, lengths, sponsors

# --- add to prepare_sequences.py ---

import re
from collections import defaultdict

def _norm_str(x):
    if pd.isna(x): return "unknown"
    return re.sub(r"\s+", "_", str(x).strip().lower())

def _first_from_pg_array(val):
    # handles NULL, plain strings, and Postgres array-like "{a,b}"
    if pd.isna(val):
        return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip() != ""]
    return _norm_str(toks[0]) if toks else "unknown"

def build_sequences_with_cats(
    csv_path="data/aact_extracted.csv",
    max_seq_len=10
):
    df = pd.read_csv(csv_path, parse_dates=["start_date", "completion_date"])
    df = df.dropna(subset=["sponsor_name", "start_date"]).copy()

    # normalize status
    df["status_norm"] = df["overall_status"].apply(_norm_status)

    # numeric features prep (same as rich)
    df["phase"] = df["phase"].fillna("N/A")
    pe = LabelEncoder()
    df["phase_enc"] = pe.fit_transform(df["phase"])

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # design categorical columns (normalize strings)
    df["allocation_norm"]       = df["allocation"].apply(_norm_str) if "allocation" in df.columns else "unknown"
    df["masking_norm"]          = df["masking"].apply(_norm_str) if "masking" in df.columns else "unknown"
    df["primary_purpose_norm"]  = df["primary_purpose"].apply(_norm_str) if "primary_purpose" in df.columns else "unknown"

    # intervention type: take first from aggregated list
    df["intv_type_norm"] = df["intervention_types"].apply(_first_from_pg_array) if "intervention_types" in df.columns else "unknown"

    # Build categorical vocabularies
    def build_vocab(series):
        vals = series.fillna("unknown").astype(str).unique().tolist()
        vals = sorted(set(_norm_str(v) for v in vals))
        idx = {v:i for i,v in enumerate(vals)}
        return idx

    vocab = {
        "allocation":      build_vocab(df["allocation_norm"]),
        "masking":         build_vocab(df["masking_norm"]),
        "primary_purpose": build_vocab(df["primary_purpose_norm"]),
        "intv_type":       build_vocab(df["intv_type_norm"]),
    }

    def map_idx(series, table):
        return series.fillna("unknown").map(lambda v: vocab[table].get(_norm_str(v), 0)).astype(int)

    df["alloc_idx"] = map_idx(df["allocation_norm"], "allocation")
    df["mask_idx"]  = map_idx(df["masking_norm"], "masking")
    df["purp_idx"]  = map_idx(df["primary_purpose_norm"], "primary_purpose")
    df["intv_idx"]  = map_idx(df["intv_type_norm"], "intv_type")

    # sort by time
    df = df.sort_values(["sponsor_name", "start_date"])

    X_num_list, X_cat_list, y_list, lengths, sponsors = [], [], [], [], []
    for sponsor, g in df.groupby("sponsor_name"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        gaps = [0.0]
        for i in range(1, len(g)):
            gaps.append(_month_diff(g.loc[i, "start_date"], g.loc[i-1, "start_date"]))
        g["gap_months"] = pd.Series(gaps).clip(lower=0.0, upper=120.0)

        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - max_seq_len):i]

            # numeric features (3): phase_enc, enroll_z, gap_months
            Xn = torch.tensor(hist[["phase_enc", "enroll_z", "gap_months"]].values, dtype=torch.float32)

            # categorical indices (4): allocation, masking, primary_purpose, intv_type
            Xc = torch.tensor(hist[["alloc_idx","mask_idx","purp_idx","intv_idx"]].values, dtype=torch.long)

            nxt = _norm_status(g.loc[i, "status_norm"])
            target = 1.0 if nxt in FAIL_STATUSES else (0.0 if nxt in SUCCESS_STATUSES else 0.0)

            X_num_list.append(Xn)
            X_cat_list.append(Xc)
            y_list.append(torch.tensor(target, dtype=torch.float32))
            lengths.append(Xn.shape[0])
            sponsors.append(sponsor)

    # pad numeric
    X_num = pad_sequence(X_num_list, batch_first=True)  # (N, T, 3)
    # pad categorical with 0s (index 0 exists in each vocab)
    max_T = max(l.size(0) for l in X_cat_list) if X_cat_list else 1
    def pad_cat(seq, T):
        pad_rows = T - seq.size(0)
        if pad_rows > 0:
            pad = torch.zeros((pad_rows, seq.size(1)), dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)
        return seq
    X_cat = torch.stack([pad_cat(xc, max_T) for xc in X_cat_list], dim=0)  # (N, T, 4)

    y = torch.stack(y_list)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    # pack vocab sizes
    vocab_sizes = {k: len(v) for k,v in vocab.items()}
    return X_num, X_cat, y, lengths, sponsors, vocab_sizes
