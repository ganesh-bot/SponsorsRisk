from typing import Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression

def fit_isotonic(y_tr: np.ndarray, p_tr: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    return iso

def apply_isotonic(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return iso.transform(p)

def calibrate_and_threshold(y_tr, p_tr, y_va, p_va) -> Tuple[IsotonicRegression, float]:
    from .metrics import best_f1_threshold
    iso = fit_isotonic(y_tr, p_tr)
    p_va_iso = apply_isotonic(iso, p_va)
    thr, _, _ = best_f1_threshold(y_va, p_va_iso)
    return iso, thr
