from __future__ import annotations

import torch
from torch import nn


class DropoutHead(nn.Module):
    """Per-gene technical-dropout classifier for SC2-medium.

    The head predicts logits, not probabilities.  Training code should supervise
    it with observed-zero/synthetic-mask labels and BCE-with-logits.
    """

    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(d_model, int(d_model * hidden_mult))
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [batch, genes, d_model], got {tuple(hidden_states.shape)}"
            )
        return self.net(hidden_states).squeeze(-1)

    @torch.no_grad()
    def probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states))
