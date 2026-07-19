from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RecoveryLossOutput:
    loss: torch.Tensor
    masked_positive_loss: torch.Tensor
    observed_nonzero_loss: torch.Tensor
    zero_regularization_loss: torch.Tensor
    n_masked_positive: int
    n_observed_nonzero: int
    n_true_zero: int


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if bool(mask.any()):
        return F.smooth_l1_loss(
            prediction[mask],
            target[mask],
            beta=beta,
            reduction="mean",
        )
    return prediction.sum() * 0.0


def single_cell_recovery_loss(
    prediction: torch.Tensor,
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    *,
    masked_positive_weight: float = 4.0,
    observed_nonzero_weight: float = 1.0,
    zero_regularization_weight: float = 0.05,
    zero_margin: float = 0.05,
    smooth_l1_beta: float = 0.2,
    zero_threshold: float = 1.0e-8,
    positive_threshold: float = 1.0e-8,
) -> RecoveryLossOutput:
    """Recovery-focused loss for synthetically corrupted single-cell batches.

    `observed_x` is the corrupted model input and `clean_target` is the same
    cell before synthetic masking. A masked positive is therefore an entry
    that is zero in `observed_x` but positive in `clean_target`.
    """
    if prediction.shape != observed_x.shape or prediction.shape != clean_target.shape:
        raise ValueError(
            "prediction, observed_x and clean_target must have identical shapes; "
            f"got {tuple(prediction.shape)}, {tuple(observed_x.shape)}, "
            f"and {tuple(clean_target.shape)}"
        )

    observed_zero = observed_x <= zero_threshold
    target_positive = clean_target > positive_threshold

    masked_positive = observed_zero & target_positive
    observed_nonzero = (~observed_zero) & target_positive
    true_zero = clean_target <= zero_threshold

    masked_positive_loss = _masked_smooth_l1(
        prediction,
        clean_target,
        masked_positive,
        beta=smooth_l1_beta,
    )
    observed_nonzero_loss = _masked_smooth_l1(
        prediction,
        clean_target,
        observed_nonzero,
        beta=smooth_l1_beta,
    )

    if bool(true_zero.any()):
        # Targets are nonnegative; penalize both positive hallucinations and
        # biologically invalid negative values outside a small tolerance.
        excess = F.relu(prediction[true_zero].abs() - float(zero_margin))
        zero_regularization_loss = torch.mean(excess.square())
    else:
        zero_regularization_loss = prediction.sum() * 0.0

    total = (
        float(masked_positive_weight) * masked_positive_loss
        + float(observed_nonzero_weight) * observed_nonzero_loss
        + float(zero_regularization_weight) * zero_regularization_loss
    )

    return RecoveryLossOutput(
        loss=total,
        masked_positive_loss=masked_positive_loss,
        observed_nonzero_loss=observed_nonzero_loss,
        zero_regularization_loss=zero_regularization_loss,
        n_masked_positive=int(masked_positive.sum().item()),
        n_observed_nonzero=int(observed_nonzero.sum().item()),
        n_true_zero=int(true_zero.sum().item()),
    )
