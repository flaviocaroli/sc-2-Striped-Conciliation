from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    counts: dict[str, int]


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


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
            beta=float(beta),
            reduction="mean",
        )
    return _zero_loss(prediction)


def _sample_negative_mask(
    negative_candidates: torch.Tensor,
    n_positive: int,
    negative_ratio: float,
) -> torch.Tensor:
    flat_candidates = negative_candidates.reshape(-1)
    candidate_indices = torch.nonzero(flat_candidates, as_tuple=False).squeeze(1)
    output = torch.zeros_like(flat_candidates, dtype=torch.bool)

    if candidate_indices.numel() == 0:
        return output.view_as(negative_candidates)

    target_count = int(round(max(1.0, float(n_positive) * float(negative_ratio))))
    target_count = min(target_count, int(candidate_indices.numel()))
    chosen = candidate_indices[
        torch.randperm(candidate_indices.numel(), device=candidate_indices.device)[:target_count]
    ]
    output[chosen] = True
    return output.view_as(negative_candidates)


def _balanced_dropout_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    eligible = positive_mask | negative_mask
    if not bool(eligible.any()):
        return _zero_loss(logits)

    labels = positive_mask[eligible].to(dtype=logits.dtype)
    selected_logits = logits[eligible]
    n_positive = int(labels.sum().item())
    n_negative = int(labels.numel() - n_positive)

    if n_positive > 0 and n_negative > 0:
        pos_weight = torch.tensor(
            float(n_negative) / float(max(n_positive, 1)),
            device=logits.device,
            dtype=logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            selected_logits,
            labels,
            pos_weight=pos_weight,
        )
    return F.binary_cross_entropy_with_logits(selected_logits, labels)


def _masked_structure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_entries: int = 2,
) -> torch.Tensor:
    masked_prediction = prediction * mask.to(dtype=prediction.dtype)
    masked_target = target * mask.to(dtype=target.dtype)
    valid = mask.sum(dim=1) >= int(min_entries)
    if not bool(valid.any()):
        return _zero_loss(prediction)
    similarity = F.cosine_similarity(
        masked_prediction[valid],
        masked_target[valid],
        dim=1,
        eps=1.0e-8,
    )
    return (1.0 - similarity).mean()


def _masked_variance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if int(mask.sum().item()) < 2:
        return _zero_loss(prediction)
    prediction_std = prediction[mask].float().std(unbiased=False)
    target_std = target[mask].float().std(unbiased=False)
    return torch.abs(
        torch.log(prediction_std + 1.0e-6)
        - torch.log(target_std + 1.0e-6)
    ).to(dtype=prediction.dtype)


def compute_sc_objective(
    outputs: Mapping[str, torch.Tensor],
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    loss_cfg: Mapping[str, Any],
) -> ObjectiveOutput:
    """Compute a selectable single-cell objective.

    Supported objective names:

    * ``legacy_mse``: whole-matrix MSE plus optional dropout BCE;
    * ``recovery``: direct masked-positive recovery with explicit zero control;
    * ``hurdle``: separate dropout classification and conditional positive-value
      estimation, with a deployed expected-repair loss.
    """
    objective_name = str(loss_cfg.get("name", "hurdle")).lower()
    beta = float(loss_cfg.get("smooth_l1_beta", 0.20))
    zero_threshold = float(loss_cfg.get("zero_threshold", 1.0e-8))

    observed_zero = observed_x <= zero_threshold
    target_positive = clean_target > zero_threshold
    masked_positive = observed_zero & target_positive
    visible_positive = (~observed_zero) & target_positive
    true_zero = clean_target <= zero_threshold

    components: dict[str, torch.Tensor] = {}
    counts = {
        "masked_positive": int(masked_positive.sum().item()),
        "visible_positive": int(visible_positive.sum().item()),
        "true_zero": int(true_zero.sum().item()),
    }

    if objective_name == "legacy_mse":
        prediction = outputs[str(loss_cfg.get("prediction_key", "reconstruction"))]
        reconstruction = F.mse_loss(prediction, clean_target)
        components["legacy_reconstruction"] = reconstruction
        total = float(loss_cfg.get("reconstruction_weight", 1.0)) * reconstruction

        if "dropout_logits" in outputs and float(loss_cfg.get("dropout_weight", 0.0)) > 0.0:
            negative_mask = _sample_negative_mask(
                true_zero,
                counts["masked_positive"],
                float(loss_cfg.get("negative_ratio", 3.0)),
            )
            dropout = _balanced_dropout_loss(
                outputs["dropout_logits"],
                masked_positive,
                negative_mask,
            )
            components["dropout"] = dropout
            total = total + float(loss_cfg.get("dropout_weight", 0.0)) * dropout
        return ObjectiveOutput(total, components, counts)

    if objective_name == "recovery":
        prediction = outputs[str(loss_cfg.get("prediction_key", "positive_value"))]
        masked_loss = _masked_smooth_l1(
            prediction,
            clean_target,
            masked_positive,
            beta=beta,
        )
        visible_loss = _masked_smooth_l1(
            prediction,
            clean_target,
            visible_positive,
            beta=beta,
        )
        zero_margin = float(loss_cfg.get("zero_margin", 0.05))
        if bool(true_zero.any()):
            excess = F.relu(prediction[true_zero].abs() - zero_margin)
            zero_loss = excess.square().mean()
        else:
            zero_loss = _zero_loss(prediction)
        structure = _masked_structure_loss(prediction, clean_target, masked_positive)
        variance = _masked_variance_loss(prediction, clean_target, masked_positive)

        components.update(
            masked_positive=masked_loss,
            visible_positive=visible_loss,
            zero_regularization=zero_loss,
            structure=structure,
            variance=variance,
        )
        total = (
            float(loss_cfg.get("masked_positive_weight", 4.0)) * masked_loss
            + float(loss_cfg.get("visible_positive_weight", 1.0)) * visible_loss
            + float(loss_cfg.get("zero_regularization_weight", 0.05)) * zero_loss
            + float(loss_cfg.get("structure_weight", 0.0)) * structure
            + float(loss_cfg.get("variance_weight", 0.0)) * variance
        )

        if "dropout_logits" in outputs and float(loss_cfg.get("dropout_weight", 0.0)) > 0.0:
            negative_mask = _sample_negative_mask(
                true_zero,
                counts["masked_positive"],
                float(loss_cfg.get("negative_ratio", 3.0)),
            )
            dropout = _balanced_dropout_loss(
                outputs["dropout_logits"],
                masked_positive,
                negative_mask,
            )
            components["dropout"] = dropout
            total = total + float(loss_cfg.get("dropout_weight", 0.0)) * dropout
        return ObjectiveOutput(total, components, counts)

    if objective_name != "hurdle":
        raise ValueError(
            "loss.name must be one of ['legacy_mse', 'recovery', 'hurdle'], "
            f"got {objective_name!r}"
        )

    required = {"positive_value", "expected_repair", "dropout_logits"}
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"Hurdle loss requires model outputs {sorted(missing)}")

    positive_value = outputs["positive_value"]
    expected_repair = outputs["expected_repair"]
    dropout_logits = outputs["dropout_logits"]

    positive_masked = _masked_smooth_l1(
        positive_value,
        clean_target,
        masked_positive,
        beta=beta,
    )
    positive_visible = _masked_smooth_l1(
        positive_value,
        clean_target,
        visible_positive,
        beta=beta,
    )
    expected_positive = _masked_smooth_l1(
        expected_repair,
        clean_target,
        masked_positive,
        beta=beta,
    )

    negative_mask = _sample_negative_mask(
        true_zero,
        counts["masked_positive"],
        float(loss_cfg.get("negative_ratio", 3.0)),
    )
    counts["sampled_negative"] = int(negative_mask.sum().item())
    dropout = _balanced_dropout_loss(
        dropout_logits,
        masked_positive,
        negative_mask,
    )
    expected_zero = (
        expected_repair[negative_mask].square().mean()
        if bool(negative_mask.any())
        else _zero_loss(expected_repair)
    )
    structure = _masked_structure_loss(
        expected_repair,
        clean_target,
        masked_positive,
    )
    variance = _masked_variance_loss(
        expected_repair,
        clean_target,
        masked_positive,
    )

    components.update(
        positive_masked=positive_masked,
        positive_visible=positive_visible,
        dropout=dropout,
        expected_positive=expected_positive,
        expected_zero=expected_zero,
        structure=structure,
        variance=variance,
    )
    total = (
        float(loss_cfg.get("positive_masked_weight", 1.0)) * positive_masked
        + float(loss_cfg.get("positive_visible_weight", 0.25)) * positive_visible
        + float(loss_cfg.get("dropout_weight", 0.50)) * dropout
        + float(loss_cfg.get("expected_positive_weight", 1.0)) * expected_positive
        + float(loss_cfg.get("expected_zero_weight", 0.10)) * expected_zero
        + float(loss_cfg.get("structure_weight", 0.20)) * structure
        + float(loss_cfg.get("variance_weight", 0.10)) * variance
    )
    return ObjectiveOutput(total, components, counts)


def compute_corruption_reconstruction_loss(
    outputs: Mapping[str, torch.Tensor],
    observed_x: torch.Tensor,
    clean_target: torch.Tensor,
    loss_cfg: Mapping[str, Any],
) -> torch.Tensor:
    prediction_key = str(loss_cfg.get("prediction_key", "reconstruction"))
    prediction = outputs[prediction_key]
    beta = float(loss_cfg.get("smooth_l1_beta", 0.20))
    corruption_threshold = float(loss_cfg.get("corruption_threshold", 1.0e-8))
    corruption_mask = (observed_x - clean_target).abs() > corruption_threshold

    if bool(corruption_mask.any()):
        loss = F.smooth_l1_loss(
            prediction[corruption_mask],
            clean_target[corruption_mask],
            beta=beta,
        )
    else:
        loss = F.smooth_l1_loss(prediction, clean_target, beta=beta)

    observed_weight = float(loss_cfg.get("observed_consistency_weight", 0.0))
    if observed_weight > 0.0:
        observed_mask = ~corruption_mask
        if bool(observed_mask.any()):
            observed_loss = F.smooth_l1_loss(
                prediction[observed_mask],
                clean_target[observed_mask],
                beta=beta,
            )
            loss = loss + observed_weight * observed_loss
    return loss
