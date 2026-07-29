from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContinuousObjectiveOutput:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    counts: dict[str, int]


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else _zero(reference)


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    alpha_t = torch.where(labels > 0.5, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def fixed_dispersion_nb_nll(
    predicted_log1p: torch.Tensor,
    raw_counts: torch.Tensor,
    library_size: torch.Tensor,
    mask: torch.Tensor,
    *,
    theta: float,
    reference_depth: float,
) -> torch.Tensor:
    """Depth-aware NB proxy using the existing positive-value head.

    This is deliberately a fixed-dispersion diagnostic, not a claim that the
    current architecture has a complete count decoder. A learned dispersion
    head should be introduced only after this proxy improves held-out count
    calibration.
    """
    if not bool(mask.any()):
        return _zero(predicted_log1p)
    depth_scale = (library_size / max(float(reference_depth), 1.0)).clamp_min(1.0e-4).unsqueeze(1)
    mean = torch.expm1(predicted_log1p.float()).clamp_min(1.0e-6) * depth_scale
    target = raw_counts.float()
    theta_tensor = torch.as_tensor(float(theta), device=mean.device, dtype=mean.dtype)
    log_prob = (
        torch.lgamma(target + theta_tensor)
        - torch.lgamma(theta_tensor)
        - torch.lgamma(target + 1.0)
        + theta_tensor * (torch.log(theta_tensor) - torch.log(theta_tensor + mean))
        + target * (torch.log(mean) - torch.log(theta_tensor + mean))
    )
    return -log_prob[mask].mean().to(dtype=predicted_log1p.dtype)


def gene_centered_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gene_mean: torch.Tensor | None,
) -> torch.Tensor:
    if gene_mean is None:
        center = target.detach().mean(dim=0, keepdim=True)
    else:
        center = gene_mean.to(device=target.device, dtype=target.dtype).view(1, -1)
    return _masked_mean(
        F.smooth_l1_loss(prediction - center, target - center, reduction="none", beta=0.20),
        mask,
        prediction,
    )


def within_cell_pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_genes_per_cell: int = 64,
    margin: float = 0.05,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row in range(prediction.shape[0]):
        indices = torch.nonzero(mask[row], as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            continue
        values = target[row, indices]
        order = torch.argsort(values)
        if order.numel() > max_genes_per_cell:
            positions = torch.linspace(
                0,
                order.numel() - 1,
                steps=max_genes_per_cell,
                device=order.device,
            ).round().long()
            order = order[positions]
        selected = indices[order]
        lower = prediction[row, selected[:-1]]
        upper = prediction[row, selected[1:]]
        target_gap = (target[row, selected[1:]] - target[row, selected[:-1]]).detach()
        valid = target_gap > 1.0e-6
        if bool(valid.any()):
            losses.append(F.softplus(float(margin) - (upper[valid] - lower[valid])).mean())
    return torch.stack(losses).mean() if losses else _zero(prediction)


def cell_structure_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_prediction = prediction * mask.to(dtype=prediction.dtype)
    masked_target = target * mask.to(dtype=target.dtype)
    valid = mask.sum(dim=1) >= 2
    if not bool(valid.any()):
        return _zero(prediction)
    return (1.0 - F.cosine_similarity(masked_prediction[valid], masked_target[valid], dim=1, eps=1.0e-8)).mean()


def variance_ratio_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) < 2:
        return _zero(prediction)
    pred_sd = prediction[mask].float().std(unbiased=False)
    target_sd = target[mask].float().std(unbiased=False)
    return torch.abs(torch.log(pred_sd + 1.0e-6) - torch.log(target_sd + 1.0e-6)).to(prediction.dtype)


def build_balanced_gate_mask(
    synthetic_positive: torch.Tensor,
    true_zero: torch.Tensor,
    *,
    negative_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_positive = int(synthetic_positive.sum().item())
    candidates = torch.nonzero(true_zero.reshape(-1), as_tuple=False).squeeze(1)
    n_negative = min(int(candidates.numel()), max(1, int(round(max(n_positive, 1) * negative_ratio))))
    negative = torch.zeros_like(true_zero, dtype=torch.bool).reshape(-1)
    if n_negative > 0:
        selected = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:n_negative]]
        negative[selected] = True
    return synthetic_positive, negative.view_as(true_zero)


def compute_continuous_objective(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    gene_mean: torch.Tensor | None = None,
) -> ContinuousObjectiveOutput:
    required = {"positive_value", "expected_repair", "dropout_logits"}
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"Continuous objective requires outputs: {sorted(missing)}")
    observed = batch["x"]
    target = batch["y"]
    counts = batch["counts"]
    synthetic_positive = batch.get("synthetic_mask", (observed <= 1.0e-8) & (target > 1.0e-8)).bool()
    true_zero = target <= float(cfg.get("zero_threshold", 1.0e-8))
    positive_value = outputs["positive_value"]
    expected = outputs["expected_repair"]
    logits = outputs["dropout_logits"]

    positive_loss = _masked_mean(
        F.smooth_l1_loss(expected, target, reduction="none", beta=float(cfg.get("smooth_l1_beta", 0.20))),
        synthetic_positive,
        expected,
    )
    gate_positive, gate_negative = build_balanced_gate_mask(
        synthetic_positive,
        true_zero,
        negative_ratio=float(cfg.get("negative_ratio", 3.0)),
    )
    gate_mask = gate_positive | gate_negative
    if bool(gate_mask.any()):
        labels = gate_positive[gate_mask].to(dtype=logits.dtype)
        gate = focal_bce_with_logits(
            logits[gate_mask],
            labels,
            gamma=float(cfg.get("focal_gamma", 2.0)),
            alpha=float(cfg.get("focal_alpha", 0.5)),
        )
    else:
        gate = _zero(logits)
    expected_zero = _masked_mean(expected.square(), gate_negative, expected)
    residual = gene_centered_residual_loss(expected, target, synthetic_positive, gene_mean)
    rank = within_cell_pairwise_rank_loss(
        expected,
        target,
        synthetic_positive,
        max_genes_per_cell=int(cfg.get("rank_max_genes", 64)),
        margin=float(cfg.get("rank_margin", 0.05)),
    )
    structure = cell_structure_loss(expected, target, synthetic_positive)
    variance = variance_ratio_loss(expected, target, synthetic_positive)
    nb = fixed_dispersion_nb_nll(
        positive_value,
        counts,
        batch["library_size"],
        synthetic_positive,
        theta=float(cfg.get("nb_theta", 10.0)),
        reference_depth=float(cfg.get("reference_depth", 10000.0)),
    )

    components = {
        "masked_count_nb": nb,
        "gene_centered_residual": residual,
        "within_cell_rank": rank,
        "cell_structure": structure,
        "repair_gate": gate,
        "expected_positive": positive_loss,
        "expected_zero": expected_zero,
        "variance": variance,
    }
    weights = dict(cfg.get("weights", {}))
    total = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    return ContinuousObjectiveOutput(
        loss=total,
        components=components,
        counts={
            "synthetic_positive": int(synthetic_positive.sum().item()),
            "gate_negative": int(gate_negative.sum().item()),
            "true_zero": int(true_zero.sum().item()),
        },
    )
