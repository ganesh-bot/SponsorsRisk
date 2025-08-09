# src/train/loops.py
from typing import Optional, Any, Dict, Tuple, Iterable
import torch
from torch.nn.utils import clip_grad_norm_


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(xx, device) for xx in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x  # leave non-tensors as-is (e.g., strings/ids)


def _extract_xy(batch, device) -> Tuple[Any, torch.Tensor]:
    """
    Accepts batches like:
      - {"X": tensor, "y": tensor}
      - {"X_num": tensor, "X_cat": tensor, "y": tensor, ...}
      - (X, y) or ([X_num, X_cat], y) or ({"X_num":..., "X_cat":...}, y)
    Returns (inputs, y) where `inputs` can be a tensor, dict, or tuple that the model can consume.
    """
    # Dict-shaped batch
    if isinstance(batch, dict):
        y = None
        for key in ("y", "label", "target"):
            if key in batch:
                y = batch[key]
                break
        if y is None:
            raise KeyError("Batch dict missing 'y'/'label'/'target'.")
        # inputs = only tensor-like entries except the label keys
        inputs = {k: v for k, v in batch.items() if k not in ("y", "label", "target")}
        # move tensors to device; keep non-tensors (e.g., strings) untouched
        inputs = _to_device(inputs, device)
        y = _to_device(y, device)
        # if there's exactly one tensor input, pass it as a single tensor (legacy models expect model(X))
        tensor_inputs = {k: v for k, v in inputs.items() if torch.is_tensor(v)}
        if len(inputs) == 1 and len(tensor_inputs) == 1:
            return next(iter(tensor_inputs.values())), y
        return inputs, y

    # Tuple/list-shaped batch
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Expected (inputs, y) but got length {len(batch)}")
        *x_parts, y = batch
        # If inputs are wrapped as a single element, unwrap
        X = x_parts[0] if len(x_parts) == 1 else tuple(x_parts)
        X = _to_device(X, device)
        y = _to_device(y, device)
        return X, y

    # Anything else: assume it's just X and y is missing
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _forward(model, inputs):
    """
    Call model with flexible inputs:
      - tensor → model(tensor)
      - dict → try model(**dict), fallback to single-tensor positional if only one tensor
      - tuple/list → model(*tuple)
    """
    if torch.is_tensor(inputs):
        return model(inputs)
    if isinstance(inputs, dict):
        try:
            return model(**inputs)
        except TypeError:
            # Fallback: if exactly one tensor value, pass it positionally
            tensor_inputs = [v for v in inputs.values() if torch.is_tensor(v)]
            if len(tensor_inputs) == 1:
                return model(tensor_inputs[0])
            raise
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    # Last resort
    return model(inputs)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_prob, all_y = [], []
    for batch in loader:
        inputs, y = _extract_xy(batch, device)
        prob = _forward(model, inputs)
        loss = criterion(prob, y.float())
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
        all_prob.append(prob.detach().cpu())
        all_y.append(y.detach().cpu())
    if n == 0:
        return 0.0, torch.empty(0), torch.empty(0)
    return total_loss / n, torch.cat(all_prob), torch.cat(all_y)


def fit(
    *,
    model,
    train_loader: Iterable,
    val_loader: Iterable,
    epochs: int,
    optimizer,
    criterion,
    scheduler: Optional[Any] = None,
    early_stopping_patience: int = 5,
    grad_clip: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    no_improve = 0

    for _ in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            inputs, y = _extract_xy(batch, device)
            prob = _forward(model, inputs)
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
            try:
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break

    return {"best_state_dict": best_state, "history": history, "best_val_loss": best_val_loss}
