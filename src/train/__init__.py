# src/train/__init__.py
from typing import Any, Dict, Optional, Tuple
import warnings
import torch

from .loops import fit as _core_fit
from .utils import infer_lengths_from_padding  # noqa: F401

_IGNORED_KEYS = {
    "batch_size", "num_workers", "shuffle", "drop_last",
    "pin_memory", "prefetch_factor", "persistent_workers",
}

def fit(model, train_loader, val_loader, *args, **kwargs) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    # Map old positionals: epochs, criterion, device
    epochs = kwargs.pop("epochs", None)
    criterion = kwargs.get("criterion", None)
    device = kwargs.get("device", "cpu")

    pos = list(args)
    if pos and epochs is None and isinstance(pos[0], int):
        epochs = pos.pop(0)
    if pos and criterion is None:
        criterion = pos.pop(0)
    if pos and isinstance(pos[0], str):
        device = pos.pop(0)
    if pos:
        warnings.warn(f"Ignoring extra positional args in fit(): {pos}", RuntimeWarning)

    if epochs is None:
        epochs = 20

    # Swallow harmless dataloader kwargs
    ignored = set(kwargs.keys()) & _IGNORED_KEYS
    for k in list(ignored):
        kwargs.pop(k, None)
    if ignored:
        warnings.warn(f"Ignoring legacy fit() args: {sorted(ignored)}", RuntimeWarning)

    # Pull trainer kwargs
    lr = kwargs.pop("lr", None)
    weight_decay = kwargs.pop("weight_decay", None)
    optimizer = kwargs.pop("optimizer", None)
    scheduler = kwargs.pop("scheduler", None)
    scheduler_name = kwargs.pop("scheduler_name", None)
    scheduler_params = kwargs.pop("scheduler_params", None)
    early_stopping_patience = kwargs.pop("early_stopping_patience", 5)
    # Map 'early_stopping' → patience if provided
    early_stopping = kwargs.pop("early_stopping", None)
    if isinstance(early_stopping, int):
        early_stopping_patience = early_stopping
    # criterion override via kw
    criterion = kwargs.pop("criterion", criterion)

    if kwargs:
        warnings.warn(f"Ignoring unknown fit() args: {sorted(kwargs.keys())}", RuntimeWarning)

    # Optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=(lr if lr is not None else 1e-3),
            weight_decay=(weight_decay if weight_decay is not None else 0.0),
        )

    # Optional scheduler by name
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
        grad_clip=kwargs.pop("grad_clip", None),
        device=device,
    )

    best = out.get("best_state_dict")
    if best:
        model.load_state_dict(best)
    return model, out

__all__ = ["fit", "infer_lengths_from_padding"]
