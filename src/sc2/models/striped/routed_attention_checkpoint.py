from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class RoutedTopKAttentionCheckpoint(nn.Module):
    """Sparse attention checkpoint with learned routing plus expression/marker priors.

    This improves over plain Top-K active-gene selection by reserving part of the
    sparse set for genes selected by a learned router.  Weak but informative marker
    genes can therefore be attended even when their raw expression is not among the
    largest values in a cell.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        top_k: int = 256,
        learned_fraction: float = 0.5,
        dropout: float = 0.1,
        expression_score_weight: float = 0.25,
        nonzero_bonus: float = 0.05,
        marker_prior_weight: float = 0.0,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= learned_fraction <= 1.0:
            raise ValueError("learned_fraction must be between 0 and 1")
        self.top_k = int(top_k)
        self.learned_fraction = float(learned_fraction)
        self.expression_score_weight = float(expression_score_weight)
        self.nonzero_bonus = float(nonzero_bonus)
        self.marker_prior_weight = float(marker_prior_weight)

        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(32, d_model // 2), 1),
        )
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.update_gate = nn.Parameter(torch.tensor(-2.0))
        if marker_prior is None:
            self.register_buffer("marker_prior", None, persistent=False)
        else:
            self.register_buffer("marker_prior", marker_prior.float(), persistent=True)

    def _route_scores(
        self,
        hidden_states: torch.Tensor,
        score_source: Optional[torch.Tensor],
        marker_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = self.router(hidden_states).squeeze(-1)
        if score_source is not None:
            if score_source.ndim == 3 and score_source.shape[-1] == 1:
                score_source = score_source.squeeze(-1)
            expr = _safe_zscore(score_source.detach().float()).to(scores.dtype)
            nonzero = (score_source.detach() > 0).to(scores.dtype)
            scores = scores + self.expression_score_weight * expr + self.nonzero_bonus * nonzero

        prior = marker_prior if marker_prior is not None else self.marker_prior
        if prior is not None and self.marker_prior_weight != 0.0:
            if prior.ndim != 1 or prior.shape[0] != scores.shape[1]:
                raise ValueError("marker_prior must have shape [n_genes]")
            scores = scores + self.marker_prior_weight * prior.to(device=scores.device, dtype=scores.dtype).view(1, -1)
        return scores

    def _select_indices(
        self,
        hidden_states: torch.Tensor,
        score_source: Optional[torch.Tensor],
        marker_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, n_genes, _ = hidden_states.shape
        k = min(self.top_k, n_genes)
        scores = self._route_scores(hidden_states, score_source, marker_prior)
        if k == n_genes:
            return torch.arange(n_genes, device=hidden_states.device).view(1, n_genes).expand(batch, n_genes)

        learned_k = int(round(k * self.learned_fraction))
        learned_k = max(0, min(k, learned_k))
        active_k = k - learned_k
        pieces = []
        if learned_k > 0:
            pieces.append(torch.topk(scores, k=learned_k, dim=1).indices)
        if active_k > 0:
            if score_source is None:
                active_scores = scores
            else:
                active_scores = score_source.squeeze(-1) if score_source.ndim == 3 else score_source
            pieces.append(torch.topk(active_scores.detach(), k=active_k, dim=1).indices)
        if not pieces:
            raise RuntimeError("empty routed attention selection")
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        score_source: Optional[torch.Tensor] = None,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [batch, genes, d_model], got {tuple(hidden_states.shape)}"
            )
        indices = self._select_indices(hidden_states, score_source, marker_prior)
        batch, _, d_model = hidden_states.shape

        gather_index = indices.to(dtype=torch.long).unsqueeze(-1).expand(batch, indices.shape[1], d_model)
        selected = hidden_states.gather(dim=1, index=gather_index)
        selected_norm = self.norm(selected)
        attended, _ = self.attn(selected_norm, selected_norm, selected_norm, need_weights=False)

        gate = torch.sigmoid(self.update_gate).to(device=attended.device, dtype=attended.dtype)
        delta = gate * self.proj(attended)
        delta = delta.to(dtype=hidden_states.dtype)

        out = hidden_states.clone()
        out.scatter_add_(dim=1, index=gather_index, src=delta)
        return out
