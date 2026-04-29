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
            "sc2_mamba_bridge.py requires mamba-ssm. "
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


class SC2MambaBridge(nn.Module):
    """
    Gene-sequence bridge model:
    - modality-specific scalar -> token input projections
    - shared gene embeddings
    - shared Mamba trunk
    - modality-specific per-token output heads
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_genes = int(n_genes)
        self.d_model = int(d_model)

        # Modality-specific scalar-to-token projections
        self.bulk_input = nn.Linear(1, d_model)
        self.sc_input = nn.Linear(1, d_model)
        self.pb_input = nn.Linear(1, d_model)

        # Shared positional/gene embedding
        self.gene_embedding = nn.Parameter(
            torch.randn(1, n_genes, d_model) * 0.02
        )

        # Modality embeddings
        self.modality_embedding = nn.ParameterDict(
            {
                "bulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "sc": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
                "pseudobulk": nn.Parameter(torch.randn(1, 1, d_model) * 0.02),
            }
        )

        self.input_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Modality-specific token-to-scalar heads
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
        """
        x: [batch, n_genes]
        returns tokens: [batch, n_genes, d_model]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape [batch, n_genes], got {tuple(x.shape)}")
        if x.shape[1] != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {x.shape[1]}")

        proj = self._input_proj(modality)
        x_tok = proj(x.unsqueeze(-1))  # [B, G, d_model]
        x_tok = x_tok + self.gene_embedding + self.modality_embedding[modality]
        x_tok = self.input_dropout(x_tok)
        return x_tok

    def forward_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.token_embed(x, modality=modality)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return h

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Returns pooled latent summary for alignment: [batch, d_model]
        """
        h = self.forward_features(x, modality=modality)
        return h.mean(dim=1)

    def decode(self, h: torch.Tensor, modality: str) -> torch.Tensor:
        """
        h: [batch, n_genes, d_model]
        returns reconstruction: [batch, n_genes]
        """
        head = self._output_head(modality)
        y = head(h).squeeze(-1)
        return y

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.forward_features(x, modality=modality)
        y = self.decode(h, modality=modality)
        return y