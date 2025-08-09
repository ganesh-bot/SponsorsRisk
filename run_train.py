# run_train.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from src.model import SponsorRiskGRU
from src.train import fit
#from src.prepare_sequences import build_sequences_rich
from src.prepare_sequences import build_sequences_rich_trends
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    # 1) Load richer sequences
    # X, y, lengths, sponsors = build_sequences_rich("data/aact_extracted.csv", max_seq_len=10)
    X, y, lengths, sponsors = build_sequences_rich_trends("data/aact_extracted.csv", max_seq_len=10)
    print("Input dims:", X.shape)  # should show (..., 9)

    # basic label check
    y_np = y.numpy()
    uniq, cnts = np.unique(y_np, return_counts=True)
    print("Label distribution:", dict(zip(uniq.tolist(), cnts.tolist())))
    if len(uniq) < 2:
        raise SystemExit("❗ Single-class labels. Check mapping or data.")

    # 2) Stratified group split (k-fold -> take first fold as val)
    tr_idx = np.load("splits/train_idx.npy")
    va_idx = np.load("splits/val_idx.npy")

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]

    # 3) Model
    input_dim = X.shape[2]  # 9 features
    model = SponsorRiskGRU(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=0.1)

    # 4) DataLoaders
    import torch
    from torch.utils.data import Dataset, DataLoader
    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X)
            self.y = torch.as_tensor(y).long()
            assert self.X.ndim == 3 and self.X.size(-1) == input_dim, f"Expected (N,L,{input_dim}), got {self.X.shape}"
        def __len__(self): return self.X.size(0)
        def __getitem__(self, i): return {"X": self.X[i], "y": self.y[i]}

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=128, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SeqDataset(X_val,   y_val),   batch_size=256, shuffle=False, num_workers=0)

    # 5) Loss (class-weighted)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = int((y_train == 1).sum().item()); neg = int((y_train == 0).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 6) Train
    trained_model, stats = fit(
        model,
        train_loader,
        val_loader,
        epochs=15,
        lr=1e-3,
        early_stopping=3,
        device=device,
        criterion=criterion,
    )

    # 7) Save
    out_path = "sponsorsrisk_gru.pt"
    torch.save(trained_model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")
    if "val_auc" in stats:
        print(f"Val AUC: {stats['val_auc']:.3f} | PR-AUC: {stats['val_prauc']:.3f} | "
            f"Best-F1: {stats['val_best_f1']:.3f} @ thr={stats['val_best_thr']:.3f}")
    else:
        print(f"Best validation loss: {stats['best_val_loss']:.4f}")