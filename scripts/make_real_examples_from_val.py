# scripts/make_real_examples_from_val.py
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

# --- make import robust when run as a script ---
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.models.combined import CombinedGRU
from src.features.prepare_sequences import build_sequences_with_cats_trends
from src.train.metrics import best_f1_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fail(msg: str):
    print(f"❌ {msg}")
    raise SystemExit(1)

def ok(msg: str):
    print(f"✅ {msg}")

def info(msg: str):
    print(f"→ {msg}")

def main():
    # --- sanity checks on inputs/artifacts ---
    data_csv = "data/aact_extracted.csv"
    tr_idx_path = "splits/train_idx.npy"
    va_idx_path = "splits/val_idx.npy"
    weights_path = "sponsorsrisk_combined.pt"  # trained by run_train_combined.py
    models_dir = "models"
    thr_json = os.path.join(models_dir, "thresholds.json")

    if not os.path.exists(data_csv): fail(f"Missing {data_csv}")
    if not os.path.exists(tr_idx_path): fail(f"Missing {tr_idx_path} (run scripts/save_splits.py)")
    if not os.path.exists(va_idx_path): fail(f"Missing {va_idx_path} (run scripts/save_splits.py)")
    if not os.path.exists(weights_path): fail(f"Missing {weights_path} (run run_train_combined.py)")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        info(f"Created {models_dir}/ (no threshold file yet; will compute from val)")

    # --- load sequences (Combined) ---
    info("Loading Combined sequences…")
    seq = build_sequences_with_cats_trends(data_csv, max_seq_len=10, verbose=False)
    # tolerate extra returned values
    Xn, Xc, y, L, sponsors, vocab_sizes, *extra = seq
    ok(f"Sequences loaded: Xn={tuple(Xn.shape)}, Xc={tuple(Xc.shape)}")

    tr_idx = np.load(tr_idx_path)
    va_idx = np.load(va_idx_path)
    ok(f"Split loaded: train={len(tr_idx)}, val={len(va_idx)}")

    # --- load model weights ---
    model = CombinedGRU(
        num_dim=Xn.shape[2], cat_vocab_sizes=vocab_sizes,
        emb_dim=16, hidden_dim=64, num_layers=1, dropout=0.1
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    ok(f"Loaded weights: {weights_path}")

    # --- predict train/val probs ---
    def batched_probs(Xn_, Xc_, L_, bs=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn_), bs):
                xn = Xn_[i:i+bs].to(DEVICE)
                xc = Xc_[i:i+bs].to(DEVICE)
                ll = L_[i:i+bs].to(DEVICE)
                p = torch.sigmoid(model(xn, xc, ll)).cpu().numpy().ravel()
                out.append(p)
        return np.concatenate(out)

    info("Scoring train/val…")
    p_tr = batched_probs(Xn[tr_idx], Xc[tr_idx], L[tr_idx])
    y_tr = y[tr_idx].numpy()
    p_va = batched_probs(Xn[va_idx], Xc[va_idx], L[va_idx])
    y_va = y[va_idx].numpy()
    ok("Scoring done.")

    # --- isotonic calibration on train, apply to val ---
    info("Fitting isotonic on train and transforming val…")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    p_va_iso = iso.transform(p_va)
    ok("Calibration done.")

    # --- threshold: use saved (paper) if present, else compute best-F1 on val (iso) ---
    if os.path.exists(thr_json):
        with open(thr_json, "r") as f:
            thr = float(json.load(f)["threshold"])
        info(f"Using saved paper threshold: {thr:.3f}")
    else:
        thr, _, _ = best_f1_threshold(y_va, p_va_iso)
        with open(thr_json, "w") as f:
            json.dump({"threshold": float(thr)}, f, indent=2)
        info(f"No thresholds.json found. Computed best-F1 thr on val: {thr:.3f} (saved to models/thresholds.json)")

    # --- pick one clear below-thr and one clear above-thr example from val ---
    info("Selecting one below-threshold and one above-threshold sponsor from val…")
    pairs = list(zip(va_idx.tolist(), p_va_iso.tolist()))
    below = sorted([p for p in pairs if p[1] < thr], key=lambda x: x[1])
    above = sorted([p for p in pairs if p[1] >= thr], key=lambda x: -x[1])
    if not below or not above:
        fail("Could not find both below and above threshold examples. Consider retraining or checking split.")
    idx0, p0 = below[0]   # most clearly below
    idx1, p1 = above[0]   # most clearly above
    ok(f"Picked: below={sponsors[idx0]} (p_iso={p0:.3f}), above={sponsors[idx1]} (p_iso={p1:.3f})")

    # --- build JSON histories from the raw CSV for these sponsors ---
    info("Reconstructing trial histories for the two sponsors…")
    df = pd.read_csv(data_csv, parse_dates=["start_date"])
    def sponsor_history(df, sponsor_name, max_n=10):
        d = df[df["sponsor_name"] == sponsor_name].copy()
        if d.empty:
            return {"sponsor_name": sponsor_name, "trials": []}
        d = d.sort_values("start_date").tail(max_n)
        def row_to_trial(r):
            def _s(v): 
                return "" if pd.isna(v) else str(v)
            def _i(v):
                try:
                    return int(v)
                except Exception:
                    return 0
            return {
                "start_date": r["start_date"].strftime("%Y-%m-%d"),
                "phase": _s(r.get("phase", "")),
                "enrollment": _i(r.get("enrollment", 0)),
                "allocation": _s(r.get("allocation", "")),
                "masking": _s(r.get("masking", "")),
                "primary_purpose": _s(r.get("primary_purpose", "")),
                "intervention_types": _s(r.get("intervention_types", "")),
                "overall_status": _s(r.get("overall_status", "")),
            }
        trials = [row_to_trial(r) for _, r in d.iterrows()]
        return {"sponsor_name": sponsor_name, "trials": trials}

    ex0 = sponsor_history(df, sponsors[idx0])
    ex1 = sponsor_history(df, sponsors[idx1])

    outdir = "data/examples"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "sample_histories.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([ex0, ex1], f, indent=2)

    ok(f"Wrote {out_path}")
    print(f"   Below-threshold example: {ex0['sponsor_name']} (p_iso={p0:.3f} < {thr:.3f})")
    print(f"   Above-threshold example: {ex1['sponsor_name']} (p_iso={p1:.3f} ≥ {thr:.3f})")

if __name__ == "__main__":
    main()
