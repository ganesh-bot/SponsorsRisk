# src/train/__init__.py
"""
Backward-compatible train API.

- Exposes fit() that accepts legacy args: lr, weight_decay, scheduler, grad_clip, etc.
- Silently ignores dataloader-related kwargs like batch_size that older code passed here.
- Returns (model, stats) to match legacy scripts.
"""
from typing import Any, Dict, Optional, Tuple
import warnings
import torch

from .loops import fit as _core_fit
from .utils import infer_lengths_from_padding  # noqa: F401


# keys we allow but don't need to forward into the core loop
_IGNORED_KEYS = {
    "batch_size",
    "num_workers",
    "shuffle",
    "drop_last",
    "pin_memory",
    "prefetch_factor",
    "persistent_workers",
    # add others here if your scripts pass them into fit()
}


def fit(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int = 20,
    # legacy-friendly args:
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion=None,
    scheduler: Optional[Any] = None,
    scheduler_name: Optional[str] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    early_stopping_patience: int = 5,
    grad_clip: Optional[float] = None,
    device: str = "cpu",
    **legacy_kwargs,  # swallow unknown legacy args (e.g., batch_size)
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    # Drop harmless extras that older runners may pass
    ignored = set(legacy_kwargs.keys()) & _IGNORED_KEYS
    if ignored:
        warnings.warn(f"Ignoring legacy fit() args: {sorted(ignored)}", RuntimeWarning)
        for k in ignored:
            legacy_kwargs.pop(k, None)
    # If anything else unexpected remains, ignore but warn once
    if legacy_kwargs:
        warnings.warn(f"Ignoring unknown fit() args: {sorted(legacy_kwargs.keys())}", RuntimeWarning)

    # Build optimizer if not supplied
    if optimizer is None:
        lr_final = lr if lr is not None else 1e-3
        wd_final = weight_decay if weight_decay is not None else 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_final, weight_decay=wd_final)

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
        # else: leave as None

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

    # Load best weights into the provided model
    best = out.get("best_state_dict")
    if best:
        model.load_state_dict(best)

    return model, out
