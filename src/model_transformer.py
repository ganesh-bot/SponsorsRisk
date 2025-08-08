# src/model_transformer.py
import torch
import torch.nn as nn
import math

def make_padding_mask(lengths, max_len):
    # True = pad positions
    idxs = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, T)
    return idxs >= lengths.unsqueeze(1)  # (N, T)

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (does not use calendar time)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        # x: (N, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class SponsorRiskTransformer(nn.Module):
    """
    Transformer encoder over trial sequences.
    Expects:
        x: (N, T, D_in)  continuous features (e.g., phase_enc, enroll_z, gap_months)
        lengths: (N,) true lengths
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        # project + add sinusoidal pe
        h = self.input_proj(x)
        h = self.posenc(h)

        # padding mask: True at pads
        pad_mask = make_padding_mask(lengths, h.size(1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # (N, T, d_model)

        # masked mean pooling over valid timesteps
        mask = (~pad_mask).unsqueeze(-1).float()  # (N, T, 1)
        h_masked = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)   # (N, 1)
        pooled = h_masked.sum(dim=1) / denom     # (N, d_model)

        logits = self.head(self.dropout(pooled)).squeeze(1)
        return logits
