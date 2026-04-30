from __future__ import annotations

import torch
import torch.nn as nn

from sc2.models.mamba_native_like import (
    BiNativeLikeMambaBlock,
    MambaLikeConfig,
    NativeLikeMambaBlock,
    clone_block,
)


class SC2NativeMambaBridge(nn.Module):
    """
    Flexible bridge wrapper for your repo.

    Supports:
    - original-like unidirectional Mamba
    - bidirectional sum
    - bidirectional gate
    - Smart Flip
    - rank ordering or fixed ordering
    - mamba1-like or mamba2-lite mixer
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        mixer_type: str = "mamba1",          # "mamba1" | "mamba2_lite"
        bidirectional: bool = True,
        merge_mode: str = "sum",             # "sum" | "gate" | "avg"
        smart_flip: bool = False,
        rank_input: bool = False,
        preserve_prefix_tokens: int = 0,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.d_model = int(d_model)
        self.rank_input = bool(rank_input)
        self.bidirectional = bool(bidirectional)

        cfg = MambaLikeConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self.bulk_value_proj = nn.Linear(1, d_model)
        self.sc_value_proj = nn.Linear(1, d_model)
        self.pb_value_proj = nn.Linear(1, d_model)

        self.gene_embedding = nn.Parameter(torch.randn(1, n_genes, d_model) * 0.02)
        self.modality_embedding = nn.ParameterDict(
            {
                "bulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "sc": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "pseudobulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            }
        )

        if bidirectional:
            block = BiNativeLikeMambaBlock(
                cfg=cfg,
                mixer_type=mixer_type,
                merge_mode=merge_mode,
                norm_type=norm_type,
                smart_flip=smart_flip,
                preserve_prefix_tokens=preserve_prefix_tokens,
                ffn_mult=2,
            )
        else:
            block = NativeLikeMambaBlock(
                cfg=cfg,
                mixer_type=mixer_type,
                norm_type=norm_type,
                ffn_mult=2,
            )

        self.blocks = clone_block(block, n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.bulk_head = nn.Linear(d_model, 1)
        self.sc_head = nn.Linear(d_model, 1)
        self.pb_head = nn.Linear(d_model, 1)

    def _value_proj(self, modality: str) -> nn.Module:
        if modality == "bulk":
            return self.bulk_value_proj
        if modality == "sc":
            return self.sc_value_proj
        if modality == "pseudobulk":
            return self.pb_value_proj
        raise ValueError(f"Unsupported modality: {modality}")

    def _output_head(self, modality: str) -> nn.Module:
        if modality == "bulk":
            return self.bulk_head
        if modality == "sc":
            return self.sc_head
        if modality == "pseudobulk":
            return self.pb_head
        raise ValueError(f"Unsupported modality: {modality}")

    def _rank_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        perm = torch.argsort(x, dim=1, descending=True, stable=True)
        x_sorted = torch.gather(x, 1, perm)
        return x_sorted, perm

    def _scatter_back(self, y_sorted: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(y_sorted)
        y.scatter_(1, perm, y_sorted)
        return y

    def token_embed(
        self,
        x: torch.Tensor,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, G = x.shape
        if G != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {G}")

        if self.rank_input:
            x_work, perm = self._rank_tokens(x)
        else:
            x_work, perm = x, None

        gene_emb = self.gene_embedding.expand(B, -1, -1)
        if perm is not None:
            gene_emb = torch.gather(
                gene_emb,
                1,
                perm.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        value_proj = self._value_proj(modality)
        value_emb = value_proj(x_work.unsqueeze(-1))

        h = gene_emb + value_emb + self.modality_embedding[modality]
        return h, perm

    def forward_features(
        self,
        x: torch.Tensor,
        modality: str,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h, perm = self.token_embed(x, modality=modality)

        for block in self.blocks:
            if self.bidirectional:
                h = block(h, valid_mask=valid_mask)
            else:
                h = block(h)

        h = self.final_norm(h)
        return h, perm

    def encode(
        self,
        x: torch.Tensor,
        modality: str,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.forward_features(x, modality=modality, valid_mask=valid_mask)
        return h.mean(dim=1)

    def decode(
        self,
        h: torch.Tensor,
        modality: str,
        perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self._output_head(modality)(h).squeeze(-1)
        if perm is not None:
            y = self._scatter_back(y, perm)
        return y

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, perm = self.forward_features(x, modality=modality, valid_mask=valid_mask)
        return self.decode(h, modality=modality, perm=perm)