from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2 as _MambaImpl
except ImportError:
    try:
        from mamba_ssm import Mamba as _MambaImpl
    except ImportError as e:
        raise ImportError(
            "sc2_hybrid_bridge.py requires mamba-ssm. "
            "Install it in the sc2 environment before training."
        ) from e


class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _MambaImpl(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        y, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        y = self.dropout(y)
        return residual + y


class SC2HybridBridge(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 4,
        attn_every: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_genes = int(n_genes)
        self.d_model = int(d_model)

        self.bulk_input = nn.Linear(1, d_model)
        self.sc_input = nn.Linear(1, d_model)
        self.pb_input = nn.Linear(1, d_model)

        self.gene_embedding = nn.Parameter(torch.randn(1, n_genes, d_model) * 0.02)
        self.modality_embedding = nn.ParameterDict(
            {
                "bulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "sc": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "pseudobulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            }
        )

        blocks: list[nn.Module] = []
        for i in range(n_layers):
            blocks.append(
                ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            )
            if attn_every > 0 and (i + 1) % attn_every == 0:
                blocks.append(
                    ResidualAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)

        self.bulk_head = nn.Linear(d_model, 1)
        self.sc_head = nn.Linear(d_model, 1)
        self.pb_head = nn.Linear(d_model, 1)

    def _input_proj(self, modality: str) -> nn.Module:
        if modality == "bulk":
            return self.bulk_input
        if modality == "sc":
            return self.sc_input
        if modality == "pseudobulk":
            return self.pb_input
        raise ValueError(f"Unsupported modality: {modality}")

    def _output_head(self, modality: str) -> nn.Module:
        if modality == "bulk":
            return self.bulk_head
        if modality == "sc":
            return self.sc_head
        if modality == "pseudobulk":
            return self.pb_head
        raise ValueError(f"Unsupported modality: {modality}")

    def token_embed(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        x_tok = self._input_proj(modality)(x.unsqueeze(-1))
        x_tok = x_tok + self.gene_embedding + self.modality_embedding[modality]
        return x_tok

    def forward_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.token_embed(x, modality=modality)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.forward_features(x, modality=modality).mean(dim=1)

    def decode(self, h: torch.Tensor, modality: str) -> torch.Tensor:
        return self._output_head(modality)(h).squeeze(-1)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.forward_features(x, modality=modality)
        return self.decode(h, modality=modality)