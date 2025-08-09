# src/train/loops.py
from typing import Optional, Dict, Any
import torch
from torch.nn.utils import clip_grad_norm_

@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_prob, all_y = [], []
    for batch in loader:
        X, y = batch["X"].to(device), batch["y"].to(device)
        prob = model(X)
        loss = criterion(prob, y.float())
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
        all_prob.append(prob.detach().cpu())
        all_y.append(y.detach().cpu())
    return total_loss / max(n, 1), torch.cat(all_prob), torch.cat(all_y)

def fit(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int,
    optimizer,
    criterion,
    scheduler: Optional[Any] = None,
    early_stopping_patience: int = 5,
    grad_clip: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Minimal, stable training loop used by run_train*.py
    Returns dict with best_state_dict and history.
    """
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            X, y = batch["X"].to(device), batch["y"].to(device)
            prob = model(X)
            loss = criterion(prob, y.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

        train_loss = total_loss / max(n, 1)
        val_loss, _, _ = _eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            # Step on val loss if it's a plateau scheduler, else per-epoch
            try:
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                break

    return {"best_state_dict": best_state, "history": history, "best_val_loss": best_val_loss}
