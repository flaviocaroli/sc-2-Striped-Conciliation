from __future__ import annotations

import torch
from torch import nn


class TopKAttentionCheckpoint(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        top_k: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, score_source: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        k = min(self.top_k, seq_len)

        if score_source is None:
            scores = x.norm(dim=-1)
        else:
            scores = score_source.abs()

        top_idx = torch.topk(scores, k=k, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])

        x_top = torch.gather(x, dim=1, index=gather_idx)
        x_top_norm = self.norm(x_top)

        attn_out, _ = self.attn(x_top_norm, x_top_norm, x_top_norm, need_weights=False)
        x_top = x_top + attn_out
        x_top = x_top + self.ffn(x_top)

        out = x.clone()
        out.scatter_(dim=1, index=gather_idx, src=x_top)
        return out