import os, json, pickle, re
import numpy as np
import pandas as pd
import torch

from src.model_combined import CombinedGRU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _norm_lower(x):
    if pd.isna(x): return "unknown"
    return str(x).strip().lower()

def _norm_underscore(x):
    if pd.isna(x): return "unknown"
    return re.sub(r"\s+", "_", str(x).strip().lower())

def _first_from_pg_array(val):
    if pd.isna(val): return "unknown"
    s = str(val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return _norm_underscore(toks[0]) if toks else "unknown"

def _compute_trends(hist: pd.DataFrame):
    h3 = hist.tail(3)
    out = {}
    fails = {"terminated","withdrawn","suspended"}
    out["prior_fail_rate_last3"] = h3["overall_status"].str.lower().isin(fails).mean() if len(h3) else 0.0
    # enroll_z assumed present
    if len(h3) >= 1:
        out["enroll_delta_last3"] = h3["enroll_z"].iloc[-1] - h3["enroll_z"].mean()
    else:
        out["enroll_delta_last3"] = 0.0
    if len(h3) >= 2:
        xs = np.arange(len(h3))
        out["enroll_slope_last3"] = float(np.polyfit(xs, h3["enroll_z"].values, 1)[0])
    else:
        out["enroll_slope_last3"] = 0.0
    out["phase_prog_last3"] = (h3["phase_enc"].iloc[-1] - h3["phase_enc"].mean()) if len(h3) else 0.0
    out["gap_mean_last3"] = h3["gap_months"].mean() if len(h3) else 0.0
    h5 = hist.tail(5)
    out["intv_diversity_last5"] = h5["intv_type_norm"].nunique() if "intv_type_norm" in hist.columns and len(h5) else 0.0
    return out

def _prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("start_date")
    # phase encoding: simple ordinal map consistent with training expectations
    # For inference we can map with a fixed order; ideally you reuse training encoders.
    # Here we create a stable map across typical AACT phases.
    phase_order = ["early phase 1","phase 1","phase 1/phase 2","phase 2","phase 2/phase 3","phase 3","phase 4","unknown","n/a"]
    phase_map = {p:i for i,p in enumerate(phase_order)}
    def canon_phase(x):
        s = str(x).lower().strip()
        s = re.sub(r"phase(\d)", r"phase \1", s)
        s = s.replace("early_phase 1","early phase 1").replace("early_phase1","early phase 1")
        return s if s in phase_map else "unknown"
    df["phase_enc"] = df.get("phase","unknown").map(canon_phase).map(phase_map).fillna(phase_map["unknown"]).astype(float)

    df["enrollment"] = pd.to_numeric(df.get("enrollment", 0.0), errors="coerce").fillna(0.0)
    df["enroll_log1p"] = np.log1p(df["enrollment"])
    mu, sd = df["enroll_log1p"].mean(), df["enroll_log1p"].std(ddof=0) or 1.0
    df["enroll_z"] = (df["enroll_log1p"] - mu) / sd

    # intervention type normalization for diversity + cat idx
    df["intervention_types"] = df.get("intervention_types", "unknown")
    df["intv_type_norm"] = df["intervention_types"].apply(_first_from_pg_array)

    # gap months
    gaps = [0.0]
    for i in range(1, len(df)):
        dt = (pd.to_datetime(df["start_date"].iloc[i]) - pd.to_datetime(df["start_date"].iloc[i-1])).days / 30.4375
        gaps.append(max(0.0, min(120.0, dt)))
    df["gap_months"] = gaps
    return df

def _map_cats_to_idx(df: pd.DataFrame, vocab_maps: dict) -> pd.DataFrame:
    df = df.copy()
    df["allocation_norm"]      = df.get("allocation","unknown").map(_norm_underscore)
    df["masking_norm"]         = df.get("masking","unknown").map(_norm_underscore)
    df["primary_purpose_norm"] = df.get("primary_purpose","unknown").map(_norm_underscore)
    df["intv_type_norm"]       = df["intv_type_norm"].map(_norm_underscore)

    def m(series, key):
        tbl = vocab_maps[key]              # string -> index
        unk = tbl.get("unknown", 0)        # be robust if 'unknown' index isn't 0
        return series.map(lambda v: tbl.get(v, unk)).astype(int)

    df["alloc_idx"] = m(df["allocation_norm"], "allocation")
    df["mask_idx"]  = m(df["masking_norm"], "masking")
    df["purp_idx"]  = m(df["primary_purpose_norm"], "primary_purpose")
    df["intv_idx"]  = m(df["intv_type_norm"], "intv_type")
    return df

def _build_sequence_rows(df_sorted: pd.DataFrame):
    rows_num, rows_cat = [], []
    for t in range(len(df_sorted)):
        h_t = df_sorted.iloc[:t+1]
        tr = _compute_trends(h_t)
        rows_num.append([
            float(df_sorted["phase_enc"].iloc[t]),
            float(df_sorted["enroll_z"].iloc[t]),
            float(df_sorted["gap_months"].iloc[t]),
            float(tr["prior_fail_rate_last3"]),
            float(tr["enroll_delta_last3"]),
            float(tr["enroll_slope_last3"]),
            float(tr["phase_prog_last3"]),
            float(tr["gap_mean_last3"]),
            float(tr["intv_diversity_last5"]),
        ])
        rows_cat.append([
            int(df_sorted["alloc_idx"].iloc[t]),
            int(df_sorted["mask_idx"].iloc[t]),
            int(df_sorted["purp_idx"].iloc[t]),
            int(df_sorted["intv_idx"].iloc[t]),
        ])
    Xn = torch.tensor(rows_num, dtype=torch.float32).unsqueeze(0)  # (1,T,9)
    Xc = torch.tensor(rows_cat, dtype=torch.long).unsqueeze(0)     # (1,T,4)
    L  = torch.tensor([len(df_sorted)], dtype=torch.int64)         # (1,)
    return Xn, Xc, L

def load_bundle(models_dir="models"):
    with open(os.path.join(models_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(models_dir, "vocab.json"), "r") as f:
        vocab_maps = json.load(f)  # dict[str, dict[str,int]]
    with open(os.path.join(models_dir, "thresholds.json"), "r") as f:
        thresholds = json.load(f)
    with open(os.path.join(models_dir, "calibrator_combined_isotonic.pkl"), "rb") as f:
        calibrator = pickle.load(f)

    cat_vocab_sizes = {k: len(v) for k, v in vocab_maps.items()}
    model = CombinedGRU(
        num_dim=meta["num_dim"],
        cat_vocab_sizes=cat_vocab_sizes,
        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1
    ).to(DEVICE)
    state = torch.load(os.path.join(models_dir, "sponsorsrisk_combined.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model, calibrator, thresholds, vocab_maps, meta

@torch.no_grad()
def predict_from_sponsor_df(df_sponsor: pd.DataFrame, model, calibrator, thresholds, vocab_maps):
    """
    df_sponsor: history of ONE sponsor up to 'now', sorted by start_date ascending.
    Required cols: start_date, phase, enrollment, allocation, masking, primary_purpose, intervention_types, overall_status (for trends).
    Returns: dict(prob_calibrated, label, threshold_used)
    """
    # ensure ordering & numeric prep
    if "start_date" in df_sponsor.columns:
        df = df_sponsor.copy()
    else:
        raise ValueError("df_sponsor must include 'start_date' column (datetime).")

    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values("start_date")
    df = _prep_numeric(df)
    df = _map_cats_to_idx(df, vocab_maps)
    Xn, Xc, L = _build_sequence_rows(df)
    Xn = Xn.to(DEVICE); Xc = Xc.to(DEVICE); L = L.to(DEVICE)

    logits = model(Xn, Xc, L)
    prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]
    prob_cal = float(calibrator.transform([prob])[0])

    thr = float(thresholds.get("Combined-9+4", 0.5))
    label = int(prob_cal >= thr)
    return {"prob_calibrated": prob_cal, "label": label, "threshold_used": thr}
