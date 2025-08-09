# src/train/__init__.py
# Re-export legacy API so existing scripts keep working.
from .loops import fit            # noqa: F401
from .utils import infer_lengths_from_padding  # noqa: F401

# Also re-export shared utilities if you have them:
# from .metrics import compute_auc_pr, best_f1_threshold  # optional
# from .calibrate import fit_isotonic, apply_isotonic     # optional
