from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from sc2.models.striped.dropout_head import DropoutHead
from sc2.models.striped.native_mamba_block import NativeStripedMambaBlock
from sc2.models.striped.rank_bin_embedding import RankBinDepthConfig, RankBinDepthEmbedding
from sc2.models.striped.routed_attention_checkpoint import RoutedTopKAttentionCheckpoint

ModalityLike = Union[str, int, torch.Tensor]


class SC2StripedMedium(nn.Module):
    """SC2-medium: stronger striped SC2 model for the lung development benchmark.

    Changes relative to SC2-mini:
    * real native/bidirectional-like Mamba blocks through NativeStripedMambaBlock;
    * rank/bin channel plus depth and modality tokens;
    * routed sparse-attention checkpoints, not only raw active-gene Top-K;
    * reconstruction head plus optional dropout-classification head.

    The default forward returns only the reconstructed matrix to stay compatible
    with existing SC2 evaluation scripts.  Set return_dict=True for dropout logits
    and latent features.
    """

    MODALITY_TO_ID = {
        "bulk": 0,
        "archs4": 0,
        "sc": 1,
        "single_cell": 1,
        "single-cell": 1,
        "pseudobulk": 2,
        "pb": 2,
        "pseudo_bulk": 2,
    }

    def __init__(
        self,
        n_genes: int = 4096,
        d_model: int = 160,
        n_mamba_blocks: int = 12,
        n_attention_checkpoints: int = 3,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 2,
        dropout: float = 0.1,
        top_k: int = 512,
        n_heads: int = 4,
        routed_learned_fraction: float = 0.5,
        expression_score_weight: float = 0.25,
        nonzero_bonus: float = 0.05,
        marker_prior_weight: float = 0.0,
        use_rank_bins: bool = True,
        n_rank_bins: int = 32,
        use_depth_token: bool = True,
        n_depth_bins: int = 8,
        use_modality_token: bool = True,
        n_modalities: int = 3,
        dropout_head: bool = True,
        bidirectional_mamba: bool = True,
        mamba_merge_mode: str = "gate",
        zero_threshold: float = 1e-8,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError("n_genes must be positive")
        if n_mamba_blocks <= 0:
            raise ValueError("n_mamba_blocks must be positive")
        self.n_genes = int(n_genes)
        self.d_model = int(d_model)
        self.n_mamba_blocks = int(n_mamba_blocks)
        self.n_attention_checkpoints = int(n_attention_checkpoints)
        self.zero_threshold = float(zero_threshold)
        self.use_modality_token = bool(use_modality_token)

        self.gene_embedding = nn.Embedding(self.n_genes, d_model)
        self.value_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.rank_depth_embedding = RankBinDepthEmbedding(
            RankBinDepthConfig(
                d_model=d_model,
                n_rank_bins=n_rank_bins,
                n_depth_bins=n_depth_bins,
                use_rank_bins=use_rank_bins,
                use_depth_token=use_depth_token,
            )
        )
        self.modality_embedding = nn.Embedding(n_modalities, d_model) if use_modality_token else None
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.mamba_blocks = nn.ModuleList(
            [
                NativeStripedMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    bidirectional=bidirectional_mamba,
                    merge_mode=mamba_merge_mode,
                )
                for _ in range(n_mamba_blocks)
            ]
        )

        self.attention_positions = self._make_attention_positions(n_mamba_blocks, n_attention_checkpoints)
        self.attention_blocks = nn.ModuleDict(
            {
                str(pos): RoutedTopKAttentionCheckpoint(
                    d_model=d_model,
                    n_heads=n_heads,
                    top_k=top_k,
                    learned_fraction=routed_learned_fraction,
                    dropout=dropout,
                    expression_score_weight=expression_score_weight,
                    nonzero_bonus=nonzero_bonus,
                    marker_prior_weight=marker_prior_weight,
                    marker_prior=marker_prior,
                )
                for pos in self.attention_positions
            }
        )

        hidden = max(64, d_model // 2)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout_head = DropoutHead(d_model=d_model, hidden_mult=2, dropout=dropout) if dropout_head else None
        self.latent_norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _make_attention_positions(n_blocks: int, n_attention: int) -> List[int]:
        if n_attention <= 0:
            return []
        raw = []
        interval = n_blocks / float(n_attention + 1)
        for idx in range(n_attention):
            pos = int(round(interval * float(idx + 1) - 1.0))
            raw.append(max(0, min(n_blocks - 1, pos)))
        positions: List[int] = []
        for pos in raw:
            if pos not in positions:
                positions.append(pos)
        return positions

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _modality_ids(self, modality: ModalityLike, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(modality, torch.Tensor):
            ids = modality.to(device=device, dtype=torch.long)
            if ids.ndim == 0:
                ids = ids.view(1).expand(batch_size)
            return ids
        if isinstance(modality, str):
            key = modality.lower()
            if key not in self.MODALITY_TO_ID:
                raise ValueError(f"Unknown modality {modality!r}; expected one of {sorted(self.MODALITY_TO_ID)}")
            return torch.full((batch_size,), self.MODALITY_TO_ID[key], dtype=torch.long, device=device)
        return torch.full((batch_size,), int(modality), dtype=torch.long, device=device)

    def encode(self, x: torch.Tensor, modality: ModalityLike = "sc") -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 2:
            raise ValueError(f"x must have shape [batch, genes], got {tuple(x.shape)}")
        if x.shape[1] != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {x.shape[1]}")

        batch_size, n_genes = x.shape
        gene_ids = torch.arange(n_genes, device=x.device, dtype=torch.long)
        gene_tokens = self.gene_embedding(gene_ids).unsqueeze(0).expand(batch_size, n_genes, self.d_model)
        zero_channel = (x.abs() <= self.zero_threshold).to(dtype=x.dtype)
        value_features = torch.stack((x, zero_channel), dim=-1)
        hidden = gene_tokens + self.value_projection(value_features)

        rank_tokens, depth_tokens = self.rank_depth_embedding(x)
        if rank_tokens is not None:
            hidden = hidden + rank_tokens
        if depth_tokens is not None:
            hidden = hidden + depth_tokens

        if self.modality_embedding is not None:
            modality_ids = self._modality_ids(modality, batch_size=batch_size, device=x.device)
            hidden = hidden + self.modality_embedding(modality_ids).unsqueeze(1)

        return self.input_dropout(self.input_norm(hidden))

    def forward_features(
        self,
        x: torch.Tensor,
        modality: ModalityLike = "sc",
        *,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.encode(x, modality=modality)
        score_source = x.squeeze(-1) if x.ndim == 3 and x.shape[-1] == 1 else x
        for idx, block in enumerate(self.mamba_blocks):
            hidden = block(hidden)
            key = str(idx)
            if key in self.attention_blocks:
                hidden = self.attention_blocks[key](
                    hidden,
                    score_source=score_source,
                    marker_prior=marker_prior,
                )
        return self.latent_norm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        modality: ModalityLike = "sc",
        *,
        return_dict: bool = False,
        return_dropout: bool = False,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
        hidden = self.forward_features(x, modality=modality, marker_prior=marker_prior)
        reconstruction = self.reconstruction_head(hidden).squeeze(-1)
        dropout_logits = self.dropout_head(hidden) if self.dropout_head is not None else None

        if return_dict:
            out: Dict[str, torch.Tensor] = {
                "reconstruction": reconstruction,
                "latent": hidden,
            }
            if dropout_logits is not None:
                out["dropout_logits"] = dropout_logits
            return out
        if return_dropout:
            return reconstruction, dropout_logits
        return reconstruction

    def forward_with_latent(
        self,
        x: torch.Tensor,
        modality: ModalityLike = "sc",
        *,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_features(x, modality=modality, marker_prior=marker_prior)
        reconstruction = self.reconstruction_head(hidden).squeeze(-1)
        pooled = hidden.mean(dim=1)
        return reconstruction, pooled

    @torch.no_grad()
    def predict_dropout_probability(self, x: torch.Tensor, modality: ModalityLike = "sc") -> Optional[torch.Tensor]:
        if self.dropout_head is None:
            return None
        hidden = self.forward_features(x, modality=modality)
        return torch.sigmoid(self.dropout_head(hidden))


def build_sc2_striped_medium_from_config(model_cfg: Dict[str, Any], n_genes: Optional[int] = None) -> SC2StripedMedium:
    """Small helper for train/eval scripts that use YAML dictionaries."""
    cfg = dict(model_cfg)
    cfg.pop("kind", None)
    if n_genes is not None:
        cfg["n_genes"] = int(n_genes)
    return SC2StripedMedium(**cfg)
