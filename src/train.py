# src/train.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

def infer_lengths_from_padding(X: torch.Tensor) -> torch.Tensor:
    """
    Infer sequence lengths by counting non-zero rows along features.
    Assumes padding rows are all-zeros (pad_sequence default).
    X: (N, T, D)
    Returns lengths: (N,)
    """
    # A row is "non-pad" if any feature != 0
    mask = (X.abs().sum(dim=2) > 0).to(torch.int64)  # (N, T)
    lengths = mask.sum(dim=1)  # (N,)
    # Safety: ensure min length = 1
    lengths[lengths == 0] = 1
    return lengths

def split_train_val(X, y, val_frac=0.2, seed=42):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    cut = int(N * (1 - val_frac))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]

def class_weight_from_labels(y: torch.Tensor):
    """
    Compute positive class weight for BCEWithLogitsLoss to mitigate imbalance.
    y in {0,1}, shape (N,)
    """
    pos = y.sum().item()
    neg = y.numel() - pos
    if pos == 0:  # avoid div by zero
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / max(pos, 1e-6), dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        xb, yb = [t.to(device) for t in batch]
        lengths = infer_lengths_from_padding(xb)
        logits = model(xb, lengths)
        loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * yb.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    eval_loss = 0.0
    for batch in tqdm(loader, desc="Eval", leave=False):
        xb, yb = [t.to(device) for t in batch]
        lengths = infer_lengths_from_padding(xb)
        logits = model(xb, lengths)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        probs = torch.sigmoid(logits)
        eval_loss += loss.item() * yb.size(0)
        all_y.append(yb.cpu())
        all_p.append(probs.cpu())

    y_true = torch.cat(all_y).numpy()
    y_prob = torch.cat(all_p).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return eval_loss / len(loader.dataset), acc, f1, auc

def make_loaders(X_train, y_train, X_val, y_val, batch_size=64):
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=SequentialSampler(val_ds))
    return train_loader, val_loader

# src/train.py (only the fit signature and start changed)
def fit(model, X_train, y_train, X_val, y_val, epochs=10, lr=1e-3, batch_size=64, device=None, early_stopping=3):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
    from tqdm import tqdm
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=SequentialSampler(val_ds))

    # imbalance handling
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor(neg / max(pos, 1e-6), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = -1.0
    best_state = None
    patience = early_stopping

    for epoch in range(1, epochs + 1):
        # ---- train loop ----
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            from .train import infer_lengths_from_padding  # reuse helper
            lengths = infer_lengths_from_padding(xb)
            logits = model(xb, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * yb.size(0)

        # ---- eval loop ----
        model.eval()
        all_y, all_p = [], []
        eval_loss = 0.0
        with torch.no_grad():
            for xb, yb in DataLoader(val_ds, batch_size=batch_size):
                xb, yb = xb.to(device), yb.to(device)
                from .train import infer_lengths_from_padding
                lengths = infer_lengths_from_padding(xb)
                logits = model(xb, lengths)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                probs = torch.sigmoid(logits)
                eval_loss += loss.item() * yb.size(0)
                all_y.append(yb.cpu())
                all_p.append(probs.cpu())

        import numpy as np
        y_true = torch.cat(all_y).numpy()
        y_prob = torch.cat(all_p).numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        print(f"[Epoch {epoch:02d}] train_loss={epoch_loss/len(train_ds):.4f} | "
              f"val_loss={eval_loss/len(val_ds):.4f} | acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

        if np.isnan(auc):
            # keep training but don't early stop on NaN AUC
            continue

        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stopping
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_auc": best_val_auc}

