from __future__ import annotations

import torch
from torch import nn

from sc2.models.striped.topk_attention_checkpoint import TopKAttentionCheckpoint


class SimpleNativeMambaLikeBlock(nn.Module):
    """
    Lightweight fallback block for SC2-mini development.

    This is not the final SC2 block. It is a stable local block that lets us
    test striped attention checkpoints while avoiding hard dependency on
    mamba-ssm. Replace with the existing native Mamba mixer once the interface
    is centralized.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        conv_y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(y))
        y = self.proj(conv_y * gate)
        return residual + self.drop(y)


class SC2StripedMini(nn.Module):
    """
    First striped SC2 prototype:
    - per-gene value embedding
    - optional rank-bin embedding
    - repeated Mamba-like blocks
    - sparse Top-K attention checkpoints
    - reconstruction head
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_mamba_blocks: int = 6,
        n_attention_checkpoints: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        top_k: int = 256,
        use_rank_bins: bool = False,
        n_rank_bins: int = 16,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.d_model = int(d_model)
        self.use_rank_bins = bool(use_rank_bins)
        self.n_rank_bins = int(n_rank_bins)

        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        if self.use_rank_bins:
            self.rank_embedding = nn.Embedding(n_rank_bins, d_model)
        else:
            self.rank_embedding = None

        blocks: list[nn.Module] = []
        attention_positions = self._attention_positions(
            n_mamba_blocks=n_mamba_blocks,
            n_attention_checkpoints=n_attention_checkpoints,
        )

        for i in range(n_mamba_blocks):
            blocks.append(
                SimpleNativeMambaLikeBlock(
                    d_model=d_model,
                    dropout=dropout,
                    kernel_size=max(3, int(d_conv) | 1),
                )
            )
            if i in attention_positions:
                blocks.append(
                    TopKAttentionCheckpoint(
                        d_model=d_model,
                        n_heads=4,
                        top_k=top_k,
                        dropout=dropout,
                    )
                )

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, 1)

    @staticmethod
    def _attention_positions(
        n_mamba_blocks: int,
        n_attention_checkpoints: int,
    ) -> set[int]:
        if n_attention_checkpoints <= 0:
            return set()

        positions = torch.linspace(
            0,
            n_mamba_blocks - 1,
            steps=n_attention_checkpoints + 2,
        )[1:-1]
        return {int(round(x.item())) for x in positions}

    def _rank_bins(self, x: torch.Tensor) -> torch.Tensor:
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1)
        bins = torch.div(
            ranks * self.n_rank_bins,
            max(1, x.shape[1]),
            rounding_mode="floor",
        )
        return torch.clamp(bins, min=0, max=self.n_rank_bins - 1).long()

    def forward_features(self, x: torch.Tensor, modality: str = "sc") -> torch.Tensor:
        bsz, n_genes = x.shape
        if n_genes != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {n_genes}")

        gene_ids = torch.arange(n_genes, device=x.device)
        gene_emb = self.gene_embedding(gene_ids).unsqueeze(0).expand(bsz, -1, -1)
        value_emb = self.value_projection(x.unsqueeze(-1))

        h = gene_emb + value_emb

        if self.rank_embedding is not None:
            rank_bins = self._rank_bins(x)
            h = h + self.rank_embedding(rank_bins)

        score_source = x

        for block in self.blocks:
            if isinstance(block, TopKAttentionCheckpoint):
                h = block(h, score_source=score_source)
            else:
                h = block(h)

        return self.final_norm(h)

    def forward(self, x: torch.Tensor, modality: str = "sc") -> torch.Tensor:
        h = self.forward_features(x, modality=modality)
        return self.reconstruction_head(h).squeeze(-1)

    def forward_with_latent(self, x: torch.Tensor, modality: str = "sc") -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_features(x, modality=modality)
        pred = self.reconstruction_head(h).squeeze(-1)
        z = h.mean(dim=1)
        return pred, z

    def encode(self, x: torch.Tensor, modality: str = "sc") -> torch.Tensor:
        h = self.forward_features(x, modality=modality)
        return h.mean(dim=1)