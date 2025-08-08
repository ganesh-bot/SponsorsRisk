# src/model_combined.py
import torch
import torch.nn as nn

class CombinedGRU(nn.Module):
    """
    Numeric + categorical embeddings -> GRU -> sigmoid head.
    X_num: (N, T, Dn)
    X_cat: (N, T, Dc_idx) with per-column vocab sizes
    """
    def __init__(
        self,
        num_dim: int,                    # numeric feature dim (Dn)
        cat_vocab_sizes: dict,           # {"allocation":V1, "masking":V2, ...}
        emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # embeddings for each categorical column (order fixed)
        self.cat_keys = ["allocation", "masking", "primary_purpose", "intv_type"]
        self.embeds = nn.ModuleDict({
            k: nn.Embedding(cat_vocab_sizes[k], emb_dim)
            for k in self.cat_keys
        })
        # input proj: numeric + 4*emb_dim
        input_dim = num_dim + emb_dim * len(self.cat_keys)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, X_num, X_cat, lengths):
        # X_num: (N, T, Dn)
        # X_cat: (N, T, 4) -> split into columns
        # embed each categorical column then concat with numeric
        cat_cols = [X_cat[:,:,i] for i in range(X_cat.size(2))]
        emb_list = []
        for key, col in zip(self.cat_keys, cat_cols):
            emb_list.append(self.embeds[key](col))  # (N, T, emb_dim)
        X = torch.cat([X_num] + emb_list, dim=2)    # (N, T, Dn + 4*emb_dim)

        packed = nn.utils.rnn.pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        last = h[-1]  # (N, hidden_dim)
        logits = self.head(self.dropout(last)).squeeze(1)
        return logits
