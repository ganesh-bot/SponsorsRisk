# scripts/smoke_train.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.prepare_sequences import build_sequences_rich_trends
from src.model import SponsorRiskGRU
from src.train import fit


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X)        # (N, L, C)
        self.y = torch.as_tensor(y).long() # (N,)
        assert self.X.ndim == 3, f"Expected (N,L,C), got {self.X.shape}"
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return {"X": self.X[i], "y": self.y[i]}


def main():
    # --- Load sequences (same function as main training) ---
    X, y, lengths, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10)
    assert X.ndim == 3, f"Expected (N,L,C), got {X.shape}"

    # --- Small, deterministic slice for speed ---
    rng = np.random.default_rng(1234)
    N = min(1024, X.shape[0])
    idx = rng.choice(X.shape[0], size=N, replace=False)
    Xs, ys = X[idx], y[idx]

    # --- Train/val split (80/20), sponsor-agnostic for smoke ---
    n_train = int(0.8 * N)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:]

    X_train, y_train = Xs[:n_train], ys[:n_train]
    X_val,   y_val   = Xs[n_train:], ys[n_train:]

    # --- DataLoaders ---
    bs_train, bs_val = 64, 128
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=bs_train, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SeqDataset(X_val,   y_val),   batch_size=bs_val,   shuffle=False, num_workers=0)

    # --- Model ---
    input_dim = X.shape[2]
    model = SponsorRiskGRU(input_dim=input_dim, hidden_dim=32, num_layers=1, dropout=0.1)

    # --- Loss (class-weighted BCEWithLogits) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)  # <— on device
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Train (single epoch, fast) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats = fit(
        model,
        train_loader,
        val_loader,
        epochs=1,            # smoke-fast
        lr=3e-3,
        early_stopping=2,    # irrelevant in 1 epoch but ok
        device=device,
        criterion=criterion, # explicit loss
    )

    # --- Checks ---
    assert stats.get("best_state_dict") is not None, "No best_state_dict returned"
    print("✅ SMOKE OK | best_val_loss:", stats.get("best_val_loss"))

    # Optional: save tiny artifact to ensure save path works
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/smoke_gru.pt")
    print("💾 saved models/smoke_gru.pt")


if __name__ == "__main__":
    main()
