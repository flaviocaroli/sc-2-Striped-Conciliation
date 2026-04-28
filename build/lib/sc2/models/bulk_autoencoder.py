from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class BulkAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims]

        encoder_layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))

        decoder_dims = list(reversed(dims))
        decoder_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out