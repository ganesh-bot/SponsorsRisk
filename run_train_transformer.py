# run_train_transformer.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from prepare_sequences import build_sequences_rich
from src.model_transformer import SponsorRiskTransformer
from src.train import fit

if __name__ == "__main__":
    # Load richer sequences (phase_enc, enroll_z, gap_months)
    X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)

    y_np = y.numpy()
    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")

    # Sponsor-safe stratified split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = np.array(sponsors)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(y_np)), y_np, groups))

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]

    model = SponsorRiskTransformer(
        input_dim=X.shape[2],   # 3
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )

    trained_model, stats = fit(
        model,
        X_train, y_train, X_val, y_val,
        epochs=15,
        lr=1e-3,
        batch_size=128,
        early_stopping=3
    )

    out_path = "sponsorsrisk_transformer.pt"
    torch.save(trained_model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    print(f"Best validation AUC: {stats['best_val_auc']:.3f}")
    