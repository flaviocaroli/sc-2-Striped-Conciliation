from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sc2.models.striped.official_mamba_block import (
    OfficialStripedMambaBlock,
)

try:
    from sc2.models.mamba_native_like import (
        BiNativeLikeMambaBlock,
        MambaLikeConfig,
        NativeLikeMambaBlock,
    )
except Exception as exc:
    BiNativeLikeMambaBlock = None
    MambaLikeConfig = None
    NativeLikeMambaBlock = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class NativeStripedMambaBlock(nn.Module):
    """Backend-selectable striped Mamba adapter."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        merge_mode: str = "gate",
        norm_type: str = "rmsnorm",
        bias: bool = False,
        backend: str = "reference",
    ) -> None:
        super().__init__()

        self.backend = str(
            backend
        ).lower()

        if self.backend == "official":
            self.block = (
                OfficialStripedMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    bidirectional=(
                        bidirectional
                    ),
                    merge_mode=merge_mode,
                    norm_type=norm_type,
                    bias=bias,
                )
            )
            return

        if self.backend != "reference":
            raise ValueError(
                "backend must be "
                "'reference' or 'official', "
                f"got {backend!r}"
            )

        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Could not import "
                "sc2.models.mamba_native_like."
            ) from _IMPORT_ERROR

        cfg = MambaLikeConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bias=bias,
        )

        if bidirectional:
            self.block = (
                self._make_bidirectional(
                    cfg,
                    merge_mode=merge_mode,
                    norm_type=norm_type,
                )
            )
        else:
            self.block = (
                self._make_unidirectional(
                    cfg,
                    norm_type=norm_type,
                )
            )

    @staticmethod
    def _make_bidirectional(
        cfg,
        *,
        merge_mode: str,
        norm_type: str,
    ) -> nn.Module:
        assert (
            BiNativeLikeMambaBlock
            is not None
        )

        attempts = (
            lambda: BiNativeLikeMambaBlock(
                cfg,
                merge_mode=merge_mode,
                norm_type=norm_type,
            ),
            lambda: BiNativeLikeMambaBlock(
                cfg,
                merge_mode=merge_mode,
            ),
            lambda: BiNativeLikeMambaBlock(
                cfg,
                merge=merge_mode,
            ),
            lambda: BiNativeLikeMambaBlock(
                cfg
            ),
        )

        last_error = None

        for make in attempts:
            try:
                return make()
            except TypeError as exc:
                last_error = exc

        raise TypeError(
            "Could not construct "
            "BiNativeLikeMambaBlock."
        ) from last_error

    @staticmethod
    def _make_unidirectional(
        cfg,
        *,
        norm_type: str,
    ) -> nn.Module:
        assert (
            NativeLikeMambaBlock
            is not None
        )

        attempts = (
            lambda: NativeLikeMambaBlock(
                cfg,
                norm_type=norm_type,
            ),
            lambda: NativeLikeMambaBlock(
                cfg
            ),
        )

        last_error = None

        for make in attempts:
            try:
                return make()
            except TypeError as exc:
                last_error = exc

        raise TypeError(
            "Could not construct "
            "NativeLikeMambaBlock."
        ) from last_error

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            try:
                return self.block(
                    hidden_states,
                    valid_mask=valid_mask,
                )
            except TypeError:
                pass

        return self.block(
            hidden_states
        )
