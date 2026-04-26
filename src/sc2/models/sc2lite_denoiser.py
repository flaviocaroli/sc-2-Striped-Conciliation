from __future__ import annotations

import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [nn.Linear(dim_in, dim_out), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SC2LiteDenoiser(nn.Module):
    """
    Modality-aware denoiser:
    - bulk adapter
    - sc adapter
    - shared encoder
    - shared decoder
    - modality-specific output heads
    """

    def __init__(
        self,
        input_dim: int,
        adapter_dim: int = 2048,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.bulk_adapter = FeedForwardBlock(input_dim, adapter_dim, dropout=dropout)
        self.sc_adapter = FeedForwardBlock(input_dim, adapter_dim, dropout=dropout)

        self.shared_encoder = nn.Sequential(
            FeedForwardBlock(adapter_dim, 1024, dropout=dropout),
            FeedForwardBlock(1024, latent_dim, dropout=dropout),
        )

        self.shared_decoder = nn.Sequential(
            FeedForwardBlock(latent_dim, 1024, dropout=dropout),
            FeedForwardBlock(1024, adapter_dim, dropout=dropout),
        )

        self.bulk_head = nn.Linear(adapter_dim, input_dim)
        self.sc_head = nn.Linear(adapter_dim, input_dim)

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "bulk":
            x = self.bulk_adapter(x)
        elif modality == "sc":
            x = self.sc_adapter(x)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return self.shared_encoder(x)

    def decode(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.shared_decoder(z)
        if modality == "bulk":
            return self.bulk_head(h)
        elif modality == "sc":
            return self.sc_head(h)
        raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        z = self.encode(x, modality=modality)
        return self.decode(z, modality=modality)