# scripts/save_splits.py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.prepare_sequences import build_sequences_rich  # 3F sequence builder

if __name__ == "__main__":
    # Build samples using the same logic GRU/Transformer use (3 features)
    X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)
    y_np = y.numpy()
    sponsors_np = np.array(sponsors)

    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(y_np)), y_np, sponsors_np))

    # save
    import os
    os.makedirs("splits", exist_ok=True)
    np.save("splits/train_idx.npy", tr_idx)
    np.save("splits/val_idx.npy", va_idx)
    print("✅ Saved splits/sponsor-safe indices:",
          f"train={tr_idx.shape[0]}", f"val={va_idx.shape[0]}")
