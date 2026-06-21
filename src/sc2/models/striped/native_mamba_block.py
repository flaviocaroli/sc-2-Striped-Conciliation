from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from sc2.models.mamba_native_like import (
        BiNativeLikeMambaBlock,
        MambaLikeConfig,
        NativeLikeMambaBlock,
    )
except Exception as exc:  # pragma: no cover - gives a clearer runtime error inside old checkouts.
    BiNativeLikeMambaBlock = None  # type: ignore[assignment]
    MambaLikeConfig = None  # type: ignore[assignment]
    NativeLikeMambaBlock = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class NativeStripedMambaBlock(nn.Module):
    """Adapter that lets striped SC2 use the repo's validated native Mamba block.

    SC2-mini used a fallback block.  This wrapper keeps the SC2-medium code stable
    even if the exact native block constructor differs slightly across branches.
    """

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
    ) -> None:
        super().__init__()
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Could not import sc2.models.mamba_native_like. Apply this patch inside the SC2 repo "
                "or keep src/ on PYTHONPATH."
            ) from _IMPORT_ERROR

        cfg = MambaLikeConfig(  # type: ignore[misc, operator]
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bias=bias,
        )
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode
        self.norm_type = norm_type

        if bidirectional:
            self.block = self._make_bidirectional(cfg, merge_mode=merge_mode, norm_type=norm_type)
        else:
            self.block = self._make_unidirectional(cfg, norm_type=norm_type)

    @staticmethod
    def _make_bidirectional(cfg, *, merge_mode: str, norm_type: str) -> nn.Module:
        assert BiNativeLikeMambaBlock is not None
        attempts = (
            lambda: BiNativeLikeMambaBlock(cfg, merge_mode=merge_mode, norm_type=norm_type),
            lambda: BiNativeLikeMambaBlock(cfg, merge_mode=merge_mode),
            lambda: BiNativeLikeMambaBlock(cfg, merge=merge_mode),
            lambda: BiNativeLikeMambaBlock(cfg),
        )
        last_error: Optional[Exception] = None
        for make in attempts:
            try:
                return make()
            except TypeError as exc:
                last_error = exc
        raise TypeError("Could not construct BiNativeLikeMambaBlock with known signatures") from last_error

    @staticmethod
    def _make_unidirectional(cfg, *, norm_type: str) -> nn.Module:
        assert NativeLikeMambaBlock is not None
        attempts = (
            lambda: NativeLikeMambaBlock(cfg, norm_type=norm_type),
            lambda: NativeLikeMambaBlock(cfg),
        )
        last_error: Optional[Exception] = None
        for make in attempts:
            try:
                return make()
            except TypeError as exc:
                last_error = exc
        raise TypeError("Could not construct NativeLikeMambaBlock with known signatures") from last_error

    def forward(self, hidden_states: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            if valid_mask is not None:
                return self.block(hidden_states, valid_mask=valid_mask)
            return self.block(hidden_states)
        except TypeError:
            return self.block(hidden_states)
