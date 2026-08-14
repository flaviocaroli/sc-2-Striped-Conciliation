from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sc2.models.mamba_native_like import (
    FeedForward,
    RMSNorm,
)


def _make_norm(
    norm_type: str,
    d_model: int,
) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)

    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)

    raise ValueError(
        f"Unsupported norm_type={norm_type!r}"
    )


class OfficialStripedMambaBlock(nn.Module):
    """Official Mamba-1 backend for SC2.

    Preserves the outer SC2 striped-block structure:

        normalize
        -> shared Mamba forward direction
        -> same shared Mamba on reversed sequence
        -> direction merge
        -> residual
        -> feed-forward residual

    Only the slow Python selective scan is replaced.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        merge_mode: str = "gate",
        norm_type: str = "rmsnorm",
        bias: bool = False,
    ) -> None:
        super().__init__()

        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError(
                "Official Mamba backend requested, "
                "but mamba_ssm could not be imported."
            ) from exc

        if merge_mode not in {
            "sum",
            "avg",
            "gate",
        }:
            raise ValueError(
                f"Unsupported merge_mode={merge_mode!r}"
            )

        if d_conv not in {2, 3, 4}:
            raise ValueError(
                "Official causal-conv1d fast path "
                "requires d_conv in {2, 3, 4}; "
                f"got {d_conv}."
            )

        self.backend = "official"
        self.bidirectional = bool(
            bidirectional
        )
        self.merge_mode = str(
            merge_mode
        )

        self.norm = _make_norm(
            norm_type,
            d_model,
        )

        self.shared_mixer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conv_bias=bias,
            bias=bias,
            use_fast_path=True,
        )

        # The surrounding SC2 model applies a generic initializer
        # recursively. Preserve official Mamba's specialized
        # internal initialization.
        for module in self.shared_mixer.modules():
            setattr(
                module,
                "_sc2_preserve_init",
                True,
            )

        self.mixer_dropout = nn.Dropout(
            dropout
        )

        if (
            self.bidirectional
            and self.merge_mode == "gate"
        ):
            self.gate_proj = nn.Linear(
                2 * d_model,
                d_model,
            )
        else:
            self.gate_proj = None

        self.ffn_norm = _make_norm(
            norm_type,
            d_model,
        )

        self.ffn = FeedForward(
            d_model,
            mult=2,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            if not bool(
                valid_mask.all().item()
            ):
                raise ValueError(
                    "Official SC2 Mamba backend "
                    "currently expects fixed-length "
                    "un-padded sequences."
                )

        normalized = self.norm(
            hidden_states
        )

        forward_output = (
            self.mixer_dropout(
                self.shared_mixer(
                    normalized
                )
            )
        )

        if not self.bidirectional:
            merged = forward_output

        else:
            reverse_input = torch.flip(
                normalized,
                dims=[1],
            )

            reverse_output = torch.flip(
                self.mixer_dropout(
                    self.shared_mixer(
                        reverse_input
                    )
                ),
                dims=[1],
            )

            if self.merge_mode == "sum":
                merged = (
                    forward_output
                    + reverse_output
                )

            elif self.merge_mode == "avg":
                merged = 0.5 * (
                    forward_output
                    + reverse_output
                )

            else:
                assert (
                    self.gate_proj
                    is not None
                )

                gate = torch.sigmoid(
                    self.gate_proj(
                        torch.cat(
                            [
                                forward_output,
                                reverse_output,
                            ],
                            dim=-1,
                        )
                    )
                )

                merged = (
                    gate
                    * forward_output
                    + (1.0 - gate)
                    * reverse_output
                )

        hidden_states = (
            hidden_states
            + merged
        )

        hidden_states = (
            hidden_states
            + self.ffn(
                self.ffn_norm(
                    hidden_states
                )
            )
        )

        return hidden_states
