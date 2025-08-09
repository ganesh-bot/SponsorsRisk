# src/model.py
import torch
import torch.nn as nn

class SponsorRiskGRU(nn.Module):
    """
    GRU-based sequence classifier with masked pooling.
    Expects inputs shaped: (batch, seq_len, input_dim)
    and a tensor of true lengths for masking.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # Pack -> GRU -> Unpack
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        # hidden: (num_layers, batch, hidden_dim); take last layer
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        logits = self.head(self.dropout(last_hidden)).squeeze(1)  # (batch,)
        return logits
