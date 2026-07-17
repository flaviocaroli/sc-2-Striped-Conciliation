from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


def _rank_ascending(values: torch.Tensor) -> torch.Tensor:
    """Return within-row ascending ranks in [0, n_genes - 1]."""
    if values.ndim != 2:
        raise ValueError(f"values must have shape [batch, genes], got {tuple(values.shape)}")
    batch, n_genes = values.shape
    order = torch.argsort(values, dim=1)
    ranks = torch.empty_like(order, dtype=torch.long)
    rank_values = torch.arange(n_genes, device=values.device, dtype=torch.long).view(1, n_genes)
    ranks.scatter_(1, order, rank_values.expand(batch, n_genes))
    return ranks


def compute_rank_bins(
    values: torch.Tensor,
    n_bins: int,
    *,
    high_expression_high_bin: bool = True,
) -> torch.Tensor:
    """Map each sample's genes to within-sample rank bins.

    The bins are deliberately rank-based rather than magnitude-based, so the token is
    robust to library size shifts and bulk/sc scale mismatch.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [batch, genes], got {tuple(values.shape)}")

    scores = values if high_expression_high_bin else -values
    ranks = _rank_ascending(scores)
    n_genes = values.shape[1]
    bins = torch.div(ranks * n_bins, max(n_genes, 1), rounding_mode="floor")
    return bins.clamp_(0, n_bins - 1).long()


def compute_depth_bins(values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute batch-relative depth/library-size bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [batch, genes], got {tuple(values.shape)}")

    batch = values.shape[0]
    if batch <= 1:
        return torch.zeros(batch, device=values.device, dtype=torch.long)
    depth = values.clamp_min(0).sum(dim=1)
    order = torch.argsort(depth, dim=0)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks.scatter_(0, order, torch.arange(batch, device=values.device, dtype=torch.long))
    bins = torch.div(ranks * n_bins, batch, rounding_mode="floor")
    return bins.clamp_(0, n_bins - 1).long()


@dataclass(frozen=True)
class RankBinDepthConfig:
    d_model: int
    n_rank_bins: int = 32
    n_depth_bins: int = 8
    use_rank_bins: bool = True
    use_depth_token: bool = True
    high_expression_high_bin: bool = True


class RankBinDepthEmbedding(nn.Module):
    """Rank/bin channel plus depth token for striped SC2 models.

    Returns one per-gene rank embedding and one per-sample depth embedding.  Both
    are added to the value/gene stream by the caller.
    """

    def __init__(self, config: RankBinDepthConfig) -> None:
        super().__init__()
        self.config = config
        if config.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.rank_embedding: Optional[nn.Embedding]
        self.depth_embedding: Optional[nn.Embedding]
        self.rank_embedding = (
            nn.Embedding(config.n_rank_bins, config.d_model) if config.use_rank_bins else None
        )
        self.depth_embedding = (
            nn.Embedding(config.n_depth_bins, config.d_model) if config.use_depth_token else None
        )

    def forward(self, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim != 2:
            raise ValueError(f"values must have shape [batch, genes], got {tuple(values.shape)}")

        rank_tokens = None
        if self.rank_embedding is not None:
            rank_bins = compute_rank_bins(
                values,
                self.config.n_rank_bins,
                high_expression_high_bin=self.config.high_expression_high_bin,
            )
            rank_tokens = self.rank_embedding(rank_bins)

        depth_tokens = None
        if self.depth_embedding is not None:
            depth_bins = compute_depth_bins(values, self.config.n_depth_bins)
            depth_tokens = self.depth_embedding(depth_bins).unsqueeze(1)

        return rank_tokens, depth_tokens
