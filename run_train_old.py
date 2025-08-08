# run_train.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.model import SponsorRiskGRU
from src.train import fit
from prepare_sequences import build_sequences

if __name__ == "__main__":
    # 1) Load sequences
    X, y, sponsors = build_sequences("data/aact_extracted.csv", max_seq_len=10)

    # If all labels are the same, warn early
    y_np = y.numpy()
    uniq, counts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), counts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Your current label mapping yields a single class. "
                         "Check run_diagnose.py and broaden FAIL_STATUSES if necessary.")

    # 2) Stratified split by label (note: this allows sponsor leakage; we’ll fix in next step)
    idx = np.arange(X.shape[0])
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_np)

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]

    # 3) Build model
    input_dim = X.shape[2]  # 2 features by default: [phase_enc, enrollment]
    model = SponsorRiskGRU(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=0.1)

    # 4) Train & Evaluate
    trained_model, stats = fit(
        model,
        X_train, y_train, X_val, y_val,   # pass explicit splits
        epochs=15,
        lr=1e-3,
        batch_size=64,
        early_stopping=3
    )

    # 5) Save model
    out_path = "sponsorsrisk_gru.pt"
    torch.save(trained_model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    print(f"Best validation AUC: {stats['best_val_auc']:.3f}")
