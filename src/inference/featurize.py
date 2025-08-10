# src/inference/featurize.py
import json, os, math, datetime as dt
import numpy as np
import torch

# ---------- helpers ----------
def _parse_date(s):
    # expect "YYYY-MM-DD"
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def _months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0

def _phase_to_num(p):
    p = (p or "").strip().lower()
    # simple encoding consistent with training intuition
    mapping = {
        "phase 1": 1.0, "phase i": 1.0,
        "phase 2": 2.0, "phase ii": 2.0,
        "phase 3": 3.0, "phase iii": 3.0,
        "phase 4": 4.0, "phase iv": 4.0,
        "phase 1/phase 2": 1.5, "phase 2/phase 3": 2.5, "phase 3/phase 4": 3.5,
    }
    return mapping.get(p, 0.0)

def _enroll_z(enrolls):
    # z-score within the sequence as a fallback (we don't have global scaler here)
    if len(enrolls) == 0:
        return []
    arr = np.asarray(enrolls, dtype=float)
    m, s = float(arr.mean()), float(arr.std() if arr.std() > 0 else 1.0)
    return [(e - m) / s for e in arr]

def _poly_slope(y):
    # slope over last <=3 points
    if len(y) < 2:
        return 0.0
    xs = np.arange(len(y), dtype=float)
    k = min(3, len(y))
    yy = np.array(y[-k:], dtype=float)
    xx = xs[-k:]
    denom = (xx**2).sum() - (xx.sum()**2)/k
    if abs(denom) < 1e-9:
        return 0.0
    slope = ((xx*yy).sum() - (xx.sum()*yy.sum())/k) / denom
    return float(slope)

def _trend_features(phases_num, enrolls, gaps, intv_tokens):
    """
    Compute per-step trend features from prefix (up to t-1).
    Returns list of 5-tuples for each t.
    """
    feats = []
    fails = []  # 1 if terminated/withdrawn else 0 for previous steps
    # Build failure flags from enrolls placeholder; we'll pass separately
    for _ in phases_num:
        fails.append(0)  # fill; we pass real fails array in caller

    for t in range(len(phases_num)):
        # consider prefix excluding current step
        idx = max(0, t-1)
        prev_idx = slice(0, t)  # up to t-1
        hist_len = t  # number of previous points

        # fail rate last 3
        last3 = fails[max(0, t-3):t]
        fail_rate = float(np.mean(last3)) if len(last3) else 0.0

        # enrollment slope last 3 (previous enrolls)
        slope = _poly_slope(enrolls[:t]) if hist_len >= 2 else 0.0

        # phase progression (current phase - mean of previous phases)
        if hist_len >= 1:
            phase_prog = float(phases_num[t] - np.mean(phases_num[:t]))
        else:
            phase_prog = 0.0

        # gap mean last 3
        gaps_prev = gaps[:t]
        gap_mean3 = float(np.mean(gaps_prev[-3:])) if len(gaps_prev) else 0.0

        # intervention diversity last 5 (unique count in previous 5)
        prev_intv = intv_tokens[:t]
        if prev_intv:
            uniq = len(set(prev_intv[-5:]))
        else:
            uniq = 0

        feats.append((fail_rate, slope, phase_prog, gap_mean3, float(uniq)))
    return feats

def _pad_2d(arr, max_len, pad_val=0.0, dtype=np.float32):
    arr = np.asarray(arr, dtype=dtype)
    n, d = arr.shape if arr.ndim == 2 else (len(arr), 1)
    out = np.full((max_len, d), pad_val, dtype=dtype)
    out[:min(n, max_len)] = arr[:min(n, max_len)]
    return out

def _pad_1d(arr, max_len, pad_val=0, dtype=np.int64):
    arr = np.asarray(arr, dtype=dtype)
    out = np.full((max_len,), pad_val, dtype=dtype)
    out[:min(len(arr), max_len)] = arr[:min(len(arr), max_len)]
    return out

def _load_vocab():
    # optional; if missing, we fall back to unknown=0
    path = os.path.join("models", "vocab.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "allocation": {"<unk>": 0},
        "masking": {"<unk>": 0},
        "primary_purpose": {"<unk>": 0},
        "intervention_types": {"<unk>": 0}
    }

def _tok(v, vocab):
    v = (v or "").strip()
    return vocab.get(v, vocab.get("<unk>", 0))

# ---------- public ----------
def featurize_history(trials, max_seq_len=10):
    """
    trials: list of dicts with keys used during training.
    returns Xn (1, T, 9), Xc (1, T, 4), L (1,)
    """
    # sort by start_date
    trials = sorted(trials, key=lambda r: r["start_date"])
    T = len(trials)
    if T == 0:
        raise ValueError("Empty trial list")

    # basic columns
    dates = [_parse_date(t["start_date"]) for t in trials]
    enrolls_raw = [int(t.get("enrollment", 0) or 0) for t in trials]
    phases_num = [_phase_to_num(t.get("phase")) for t in trials]
    alloc = [t.get("allocation", "") for t in trials]
    mask  = [t.get("masking", "") for t in trials]
    purpose = [t.get("primary_purpose", "") for t in trials]
    intv = [t.get("intervention_types", "") for t in trials]

    # gaps in months (first gap=0)
    gaps = [0.0] + [_months_between(dates[i-1], dates[i]) for i in range(1, T)]

    # enrollment z-score within sequence (fallback)
    enroll_z = _enroll_z(enrolls_raw)

    # failure flags from overall_status for trend calc
    status = [(t.get("overall_status","") or "").strip().lower() for t in trials]
    fail_flag = [1 if ("terminated" in s or "withdrawn" in s) else 0 for s in status]

    # intervention token (simplify: use the whole string)
    intv_tok = [s.strip("{} ") for s in intv]

    # trend features per step (computed from prefix)
    # temporarily set internal fails; we recompute fail rate from real flags below
    trend = _trend_features(phases_num, enrolls_raw, gaps, intv_tok)
    # replace fail_rate with real from last3 flags
    trend2 = []
    for t in range(T):
        last3f = fail_flag[max(0, t-3):t]
        fail_rate = float(np.mean(last3f)) if len(last3f) else 0.0
        _, slope, phase_prog, gap_mean3, uniq = trend[t]
        trend2.append((fail_rate, slope, phase_prog, gap_mean3, uniq))

    # numeric (9): [phase_enc, enroll_z, gap_months] + 5 trends + (pad one extra if you had 9 incl. something else)
    # Based on your training plan we use exactly 9: 3 base + 5 trend + 1 placeholder (0.0) if needed.
    base_num = np.stack([phases_num, enroll_z, gaps], axis=1)
    trend_arr = np.array(trend2, dtype=np.float32)  # (T,5)
    # If your model expects 9 numeric chans, concat base(3)+trend(5)+zeros(1)
    if base_num.shape[1] + trend_arr.shape[1] == 8:
        placeholder = np.zeros((T,1), dtype=np.float32)
        num = np.concatenate([base_num.astype(np.float32), trend_arr, placeholder], axis=1)
    else:
        num = np.concatenate([base_num.astype(np.float32), trend_arr], axis=1)  # (T, 8) or (T, >8)
    # if still not 9, pad/truncate channels
    if num.shape[1] < 9:
        padc = 9 - num.shape[1]
        num = np.concatenate([num, np.zeros((T,padc), dtype=np.float32)], axis=1)
    elif num.shape[1] > 9:
        num = num[:, :9]

    # categoricals: map via vocab (or unknown=0)
    vocab = _load_vocab()
    alc_map = vocab.get("allocation", {"<unk>":0})
    msk_map = vocab.get("masking", {"<unk>":0})
    pur_map = vocab.get("primary_purpose", {"<unk>":0})
    itp_map = vocab.get("intervention_types", {"<unk>":0})

    cat = np.stack([
        np.array([_tok(a, alc_map) for a in alloc], dtype=np.int64),
        np.array([_tok(m, msk_map) for m in mask], dtype=np.int64),
        np.array([_tok(p, pur_map) for p in purpose], dtype=np.int64),
        np.array([_tok(i.strip("{} "), itp_map) for i in intv], dtype=np.int64),
    ], axis=1)  # (T,4)

    # pad to max_seq_len
    num_pad = _pad_2d(num, max_seq_len, pad_val=0.0, dtype=np.float32)
    cat_pad = _pad_2d(cat, max_seq_len, pad_val=0, dtype=np.int64)
    L = np.array([min(T, max_seq_len)], dtype=np.int64)

    # batch dim = 1
    Xn = torch.tensor(num_pad[None, ...], dtype=torch.float32)
    Xc = torch.tensor(cat_pad[None, ...], dtype=torch.long)
    L  = torch.tensor(L, dtype=torch.long)
    return Xn, Xc, L
