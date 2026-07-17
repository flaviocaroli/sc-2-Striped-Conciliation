from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DropoutLossOutput:
    loss: torch.Tensor
    n_supervised: int
    n_positive: int
    positive_rate: float


def make_synthetic_dropout_labels(
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    *,
    zero_threshold: float = 1e-8,
    positive_threshold: float = 1e-8,
    only_observed_zeros: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dropout labels from masked/noisy input and clean target.

    Label 1 means: the model sees zero in observed_x, but clean_target says the
    gene should be positive.  That is the synthetic technical-dropout event.
    """
    if observed_x.shape != clean_target.shape:
        raise ValueError(
            f"observed_x and clean_target must have same shape, got {tuple(observed_x.shape)} and {tuple(clean_target.shape)}"
        )
    observed_zero = observed_x <= zero_threshold
    target_positive = clean_target > positive_threshold
    labels = (observed_zero & target_positive).to(dtype=clean_target.dtype)
    supervised_mask = observed_zero if only_observed_zeros else torch.ones_like(observed_zero, dtype=torch.bool)
    return labels, supervised_mask


def dropout_bce_loss(
    dropout_logits: Optional[torch.Tensor],
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    *,
    zero_threshold: float = 1e-8,
    positive_threshold: float = 1e-8,
    only_observed_zeros: bool = True,
    max_pos_weight: float = 20.0,
) -> DropoutLossOutput:
    """BCE loss for the SC2 dropout head with automatic class balancing."""
    if dropout_logits is None:
        zero = clean_target.sum() * 0.0
        return DropoutLossOutput(loss=zero, n_supervised=0, n_positive=0, positive_rate=0.0)
    if dropout_logits.shape != clean_target.shape:
        raise ValueError(
            f"dropout_logits and clean_target must have same shape, got {tuple(dropout_logits.shape)} and {tuple(clean_target.shape)}"
        )
    labels, supervised_mask = make_synthetic_dropout_labels(
        observed_x,
        clean_target,
        zero_threshold=zero_threshold,
        positive_threshold=positive_threshold,
        only_observed_zeros=only_observed_zeros,
    )
    if supervised_mask.sum().item() == 0:
        zero = dropout_logits.sum() * 0.0
        return DropoutLossOutput(loss=zero, n_supervised=0, n_positive=0, positive_rate=0.0)

    selected_logits = dropout_logits[supervised_mask]
    selected_labels = labels[supervised_mask]
    n_positive = int(selected_labels.sum().item())
    n_total = int(selected_labels.numel())
    n_negative = max(n_total - n_positive, 1)
    if n_positive > 0:
        pos_weight = min(float(n_negative) / float(n_positive), max_pos_weight)
        pos_weight_tensor = torch.tensor(pos_weight, device=selected_logits.device, dtype=selected_logits.dtype)
    else:
        pos_weight_tensor = torch.tensor(1.0, device=selected_logits.device, dtype=selected_logits.dtype)

    loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_labels.to(dtype=selected_logits.dtype),
        pos_weight=pos_weight_tensor,
    )
    return DropoutLossOutput(
        loss=loss,
        n_supervised=n_total,
        n_positive=n_positive,
        positive_rate=float(n_positive) / float(max(n_total, 1)),
    )


def zero_false_positive_penalty(
    reconstruction: torch.Tensor,
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    *,
    zero_threshold: float = 1e-8,
    positive_threshold: float = 1e-8,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize filling zeros that are likely true zeros, not synthetic dropouts."""
    true_zero_mask = (observed_x <= zero_threshold) & (clean_target <= positive_threshold)
    if true_zero_mask.sum().item() == 0:
        return reconstruction.sum() * 0.0
    excess = (reconstruction[true_zero_mask] - margin).clamp_min(0.0)
    return torch.mean(excess.square())
