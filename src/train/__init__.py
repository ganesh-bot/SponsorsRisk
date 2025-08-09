# src/train/__init__.py
from typing import Any, Dict, Optional, Tuple, Iterable
import warnings
import torch

from .loops import fit as _core_fit
from .utils import infer_lengths_from_padding  # noqa: F401

_IGNORED_KEYS = {
    "batch_size", "num_workers", "shuffle", "drop_last",
    "pin_memory", "prefetch_factor", "persistent_workers",
}


def _pair_loader_with_labels(loader: Iterable, y_tensor: torch.Tensor):
    """
    Wrap a loader that yields X-only batches (tensor or [tensor]) and
    attach matching y chunks in sequence order. Assumes loader iteration
    order matches y_tensor slicing (i.e., no shuffling).
    """
    if not torch.is_tensor(y_tensor):
        y_tensor = torch.as_tensor(y_tensor)

    def _gen():
        offset = 0
        for batch in loader:
            # Handle plain tensor batch
            if torch.is_tensor(batch):
                X = batch
            # Handle 1-tuple/list like (X,) or [X]
            elif isinstance(batch, (list, tuple)) and len(batch) == 1 and torch.is_tensor(batch[0]):
                X = batch[0]
            # If it's already (X, y) or dict, just yield as-is
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                yield batch
                continue
            elif isinstance(batch, dict):
                # If user already supplies y, don't override
                if "y" in batch or "label" in batch or "target" in batch:
                    yield batch
                    continue
                X = batch.get("X", None)
                if X is None:
                    # can't infer; pass through unchanged
                    yield batch
                    continue
            else:
                # unknown shape; pass through
                yield batch
                continue

            bs = int(X.size(0))
            y_chunk = y_tensor[offset:offset + bs]
            offset += bs
            # standardize to dict
            yield {"X": X, "y": y_chunk}

    return _gen()


def fit(model, train_loader, val_loader, *args, **kwargs) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    # --- map legacy positionals: epochs, criterion, device, [y_train], [y_val] ---
    epochs = kwargs.pop("epochs", None)
    criterion = kwargs.get("criterion", None)
    device = kwargs.get("device", "cpu")

    pos = list(args)
    # 4th positional → epochs
    if pos and epochs is None and isinstance(pos[0], int):
        epochs = pos.pop(0)
    # 5th positional → criterion (callable) or labels tensor
    y_train_extra = None
    if pos:
        if callable(pos[0]) or isinstance(pos[0], torch.nn.Module):
            if criterion is None:
                criterion = pos.pop(0)
        elif torch.is_tensor(pos[0]):
            y_train_extra = pos.pop(0)
    # 6th positional → device or y_val
    y_val_extra = None
    if pos:
        if isinstance(pos[0], str):
            device = pos.pop(0)
        elif torch.is_tensor(pos[0]):
            y_val_extra = pos.pop(0)
    if pos:
        warnings.warn(f"Ignoring extra positional args in fit(): {pos}", RuntimeWarning)

    if epochs is None:
        epochs = 20


    # swallow harmless dataloader kwargs
    ignored = set(kwargs.keys()) & _IGNORED_KEYS
    for k in list(ignored):
        kwargs.pop(k, None)
    if ignored:
        warnings.warn(f"Ignoring legacy fit() args: {sorted(ignored)}", RuntimeWarning)

    # trainer kwargs
    lr = kwargs.pop("lr", None)
    weight_decay = kwargs.pop("weight_decay", None)
    optimizer = kwargs.pop("optimizer", None)
    scheduler = kwargs.pop("scheduler", None)
    scheduler_name = kwargs.pop("scheduler_name", None)
    scheduler_params = kwargs.pop("scheduler_params", None)
    early_stopping_patience = kwargs.pop("early_stopping_patience", 5)
    early_stopping = kwargs.pop("early_stopping", None)
    if isinstance(early_stopping, int):
        early_stopping_patience = early_stopping
    grad_clip = kwargs.pop("grad_clip", None)
    # allow kw override for criterion
    # before parsing leftover kwargs
    device = kwargs.pop("device", device) if "device" in kwargs else device  # accept kw
    # ...
    criterion = kwargs.pop("criterion", criterion)
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    if kwargs:
        warnings.warn(f"Ignoring unknown fit() args: {sorted(kwargs.keys())}", RuntimeWarning)

    # If label tensors were passed positionally, wrap loaders to attach y
    if y_train_extra is not None:
        train_loader = _pair_loader_with_labels(train_loader, y_train_extra)
        # IMPORTANT: this assumes your DataLoader isn't shuffling.
        # If it is, please disable shuffle or pass (X, y) via the Dataset itself.
    if y_val_extra is not None:
        val_loader = _pair_loader_with_labels(val_loader, y_val_extra)

    # optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=(lr if lr is not None else 1e-3),
            weight_decay=(weight_decay if weight_decay is not None else 0.0),
        )

    # optional scheduler by name
    if scheduler is None and scheduler_name:
        p = scheduler_params or {}
        name = scheduler_name.lower()
        if name in ("plateau", "reduceonplateau", "rop"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=p.get("mode", "min"),
                factor=p.get("factor", 0.5),
                patience=p.get("patience", 3),
                min_lr=p.get("min_lr", 1e-6),
                verbose=p.get("verbose", False),
            )
        elif name in ("step", "steplr"):
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=p.get("step_size", 5),
                gamma=p.get("gamma", 0.5),
                verbose=p.get("verbose", False),
            )

    out = _core_fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        grad_clip=grad_clip,
        device=device,
    )

    best = out.get("best_state_dict")
    if best:
        model.load_state_dict(best)
    return model, out

__all__ = ["fit", "infer_lengths_from_padding"]
