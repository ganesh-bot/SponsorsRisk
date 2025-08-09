from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, accuracy_score
)

def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    auc = roc_auc_score(y_true, y_prob)
    pr  = average_precision_score(y_true, y_prob)
    return float(auc), float(pr)

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.r_[0.0, t]
    f1 = 2 * (p*r) / np.maximum(p + r, 1e-9)
    k = int(np.nanargmax(f1))
    thr = float(t[k])
    y_hat = (y_prob >= thr).astype(int)
    return thr, float(accuracy_score(y_true, y_hat)), float(f1_score(y_true, y_hat))

def slice_by_histlen(hist_lens, y_true, y_prob) -> Dict[str, Dict[str, float]]:
    out = {}
    lens = np.asarray(hist_lens)
    for name, mask in {
        "L<=3": lens <= 3,
        "4<=L<=7": (lens >= 4) & (lens <= 7),
        "L>=8": lens >= 8,
    }.items():
        if mask.sum() < 10:  # too small
            continue
        auc, pr = compute_auc_pr(y_true[mask], y_prob[mask])
        out[name] = {"auc": auc, "prauc": pr, "n": int(mask.sum())}
    return out
