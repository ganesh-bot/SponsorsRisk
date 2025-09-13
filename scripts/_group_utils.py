# scripts/_group_utils.py
import numpy as np

SPONSOR_TYPE_TO_ID = {
    "Industry": 0,
    "NIH": 1,
    "Academic": 2,
    "Other": 3,
}

def map_sponsor_type(series_like):
    def norm(x):
        if x is None: return "Other"
        s = str(x).strip().lower()
        if "industry" in s: return "Industry"
        if s in ("nih", "national institutes of health"): return "NIH"
        if "university" in s or "academic" in s: return "Academic"
        return "Other"
    arr = [SPONSOR_TYPE_TO_ID[norm(x)] for x in series_like]
    return np.asarray(arr, dtype=np.int64)
