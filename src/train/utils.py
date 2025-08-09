# src/train/utils.py
import torch

def infer_lengths_from_padding(x, pad_value=0):
    """
    Infer sequence lengths from padding.
    Works for (B, L) or (B, L, C) tensors.
    pad_value=0 assumed for zero-padded sequences by default.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 3:
        # (B, L, C): a timestep is padding if *all* channels are pad_value (or all zeros)
        if pad_value == 0:
            step_is_real = x.abs().sum(dim=-1) != 0
        else:
            step_is_real = (x != pad_value).any(dim=-1)
    elif x.ndim == 2:
        step_is_real = x != pad_value
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

    lengths = step_is_real.sum(dim=1)
    return lengths
