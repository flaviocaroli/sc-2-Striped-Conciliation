from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from sc2.models.striped.sc2_striped_medium import (
    ModalityLike,
    SC2StripedMedium,
)


class SC2StripedFull(SC2StripedMedium):
    """Configurable 20+4 striped SC2 model.

    The backbone is the existing SC2 striped implementation, configured with
    20 native Mamba blocks and 4 routed sparse-attention checkpoints by
    default. This wrapper adds:

    * optional activation checkpointing for lower memory use;
    * a non-negative conditional positive-value head;
    * explicit dropout probability and expected-repair outputs;
    * selectable deployed reconstruction behavior.

    ``reconstruction_mode`` choices:

    * ``raw``: unconstrained raw head output;
    * ``positive``: activated positive-value output;
    * ``expected``: sigmoid(dropout_logits) * positive_value;
    * ``preserve_observed``: keep observed nonzeros unchanged and use expected
      repair only where the input is zero.
    """

    def __init__(
        self,
        n_genes: int = 4096,
        d_model: int = 128,
        n_mamba_blocks: int = 20,
        n_attention_checkpoints: int = 4,
        d_state: int = 4,
        d_conv: int = 5,
        expand: int = 1,
        dropout: float = 0.10,
        top_k: int = 256,
        n_heads: int = 4,
        routed_learned_fraction: float = 0.50,
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
        zero_threshold: float = 1.0e-8,
        marker_prior: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = True,
        positive_activation: str = "softplus",
        softplus_beta: float = 1.0,
        positive_output_bias_init: Optional[float] = -2.0,
        reconstruction_mode: str = "preserve_observed",
    ) -> None:
        super().__init__(
            n_genes=n_genes,
            d_model=d_model,
            n_mamba_blocks=n_mamba_blocks,
            n_attention_checkpoints=n_attention_checkpoints,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            top_k=top_k,
            n_heads=n_heads,
            routed_learned_fraction=routed_learned_fraction,
            expression_score_weight=expression_score_weight,
            nonzero_bonus=nonzero_bonus,
            marker_prior_weight=marker_prior_weight,
            use_rank_bins=use_rank_bins,
            n_rank_bins=n_rank_bins,
            use_depth_token=use_depth_token,
            n_depth_bins=n_depth_bins,
            use_modality_token=use_modality_token,
            n_modalities=n_modalities,
            dropout_head=dropout_head,
            bidirectional_mamba=bidirectional_mamba,
            mamba_merge_mode=mamba_merge_mode,
            zero_threshold=zero_threshold,
            marker_prior=marker_prior,
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.positive_activation = str(positive_activation).lower()
        self.softplus_beta = float(softplus_beta)
        self.reconstruction_mode = str(reconstruction_mode).lower()

        valid_activations = {"identity", "relu", "softplus"}
        if self.positive_activation not in valid_activations:
            raise ValueError(
                f"positive_activation must be one of {sorted(valid_activations)}, "
                f"got {self.positive_activation!r}"
            )

        valid_modes = {"raw", "positive", "expected", "preserve_observed"}
        if self.reconstruction_mode not in valid_modes:
            raise ValueError(
                f"reconstruction_mode must be one of {sorted(valid_modes)}, "
                f"got {self.reconstruction_mode!r}"
            )
        if self.reconstruction_mode in {"expected", "preserve_observed"} and self.dropout_head is None:
            raise ValueError(
                "reconstruction_mode requires dropout_head=true because expected repair "
                "uses dropout probabilities"
            )

        if positive_output_bias_init is not None:
            final_linear = self._last_linear(self.reconstruction_head)
            if final_linear.bias is not None:
                nn.init.constant_(final_linear.bias, float(positive_output_bias_init))

    @staticmethod
    def _last_linear(module: nn.Module) -> nn.Linear:
        linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not linears:
            raise RuntimeError("reconstruction_head contains no Linear layer")
        return linears[-1]

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _activate_positive(self, raw: torch.Tensor) -> torch.Tensor:
        if self.positive_activation == "identity":
            return raw
        if self.positive_activation == "relu":
            return F.relu(raw)
        return F.softplus(raw, beta=self.softplus_beta)

    def forward_features(
        self,
        x: torch.Tensor,
        modality: ModalityLike = "sc",
        *,
        marker_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (self.gradient_checkpointing and self.training):
            return super().forward_features(x, modality=modality, marker_prior=marker_prior)

        hidden = self.encode(x, modality=modality)
        score_source = x.squeeze(-1) if x.ndim == 3 and x.shape[-1] == 1 else x

        for index, block in enumerate(self.mamba_blocks):
            hidden = checkpoint(
                lambda tensor, current_block=block: current_block(tensor),
                hidden,
                use_reentrant=False,
            )
            key = str(index)
            if key in self.attention_blocks:
                attention_block = self.attention_blocks[key]
                hidden = checkpoint(
                    lambda tensor, current_block=attention_block: current_block(
                        tensor,
                        score_source=score_source,
                        marker_prior=marker_prior,
                    ),
                    hidden,
                    use_reentrant=False,
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
        reconstruction_mode: Optional[str] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Dict[str, torch.Tensor],
    ]:
        observed = x.squeeze(-1) if x.ndim == 3 and x.shape[-1] == 1 else x
        hidden = self.forward_features(observed, modality=modality, marker_prior=marker_prior)
        raw_value = self.reconstruction_head(hidden).squeeze(-1)
        positive_value = self._activate_positive(raw_value)
        dropout_logits = self.dropout_head(hidden) if self.dropout_head is not None else None

        if dropout_logits is None:
            dropout_probability = torch.ones_like(positive_value)
        else:
            dropout_probability = torch.sigmoid(dropout_logits)

        expected_repair = dropout_probability * positive_value
        mode = str(reconstruction_mode or self.reconstruction_mode).lower()

        if mode == "raw":
            reconstruction = raw_value
        elif mode == "positive":
            reconstruction = positive_value
        elif mode == "expected":
            reconstruction = expected_repair
        elif mode == "preserve_observed":
            reconstruction = torch.where(
                observed.abs() > self.zero_threshold,
                observed,
                expected_repair,
            )
        else:
            raise ValueError(f"Unknown reconstruction_mode={mode!r}")

        if return_dict:
            output: Dict[str, torch.Tensor] = {
                "reconstruction": reconstruction,
                "raw_value": raw_value,
                "positive_value": positive_value,
                "dropout_probability": dropout_probability,
                "expected_repair": expected_repair,
                "latent": hidden,
            }
            if dropout_logits is not None:
                output["dropout_logits"] = dropout_logits
            return output

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
        output = self.forward(
            x,
            modality=modality,
            return_dict=True,
            marker_prior=marker_prior,
        )
        assert isinstance(output, dict)
        pooled = output["latent"].mean(dim=1)
        return output["reconstruction"], pooled



def build_sc2_striped_full_from_config(
    model_cfg: Dict[str, Any],
    n_genes: Optional[int] = None,
) -> SC2StripedFull:
    cfg = dict(model_cfg)
    cfg.pop("kind", None)
    if n_genes is not None:
        cfg["n_genes"] = int(n_genes)
    return SC2StripedFull(**cfg)
