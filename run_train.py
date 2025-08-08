# run_train.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from src.model import SponsorRiskGRU
from src.train import fit
from prepare_sequences import build_sequences_rich

if __name__ == "__main__":
    # 1) Load richer sequences
    X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)

    # basic label check
    y_np = y.numpy()
    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")

    # 2) Stratified group split (k-fold -> take first fold as val)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = np.array(sponsors)
    split_iter = sgkf.split(np.zeros(len(y_np)), y_np, groups)
    train_idx, val_idx = next(split_iter)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # 3) Model
    input_dim = X.shape[2]  # now 3 features: [phase_enc, enroll_z, gap_months]
    model = SponsorRiskGRU(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=0.1)

    # 4) Fit (uses explicit splits)
    trained_model, stats = fit(
        model,
        X_train, y_train, X_val, y_val,
        epochs=15,
        lr=1e-3,
        batch_size=128,
        early_stopping=3
    )

    # 5) Save
    out_path = "sponsorsrisk_gru.pt"
    torch.save(trained_model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    print(f"Best validation AUC: {stats['best_val_auc']:.3f}")
