from __future__ import annotations

import torch
import torch.nn.functional as F


def corruption_mask_from_xy(x_corrupt: torch.Tensor, y_clean: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Infer which positions were corrupted by comparing corrupted input x to clean target y.
    This works well for the current setup where corruption is mainly masking-to-zero
    and optional noise is zero or small.
    """
    return (torch.abs(x_corrupt - y_clean) > eps).float()


def weighted_masked_mse(
    pred: torch.Tensor,
    y_clean: torch.Tensor,
    x_corrupt: torch.Tensor,
    masked_position_weight: float = 4.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Loss = MSE over all positions + extra-weighted MSE on corrupted positions.
    """
    loss_all = F.mse_loss(pred, y_clean)

    mask = corruption_mask_from_xy(x_corrupt=x_corrupt, y_clean=y_clean, eps=eps)
    n_masked = mask.sum()

    if n_masked.item() > 0:
        diff2 = (pred - y_clean) ** 2
        loss_masked = (diff2 * mask).sum() / n_masked
    else:
        loss_masked = torch.zeros((), device=pred.device, dtype=pred.dtype)

    loss_total = loss_all + masked_position_weight * loss_masked

    stats = {
        "loss_all": float(loss_all.detach().cpu().item()),
        "loss_masked": float(loss_masked.detach().cpu().item()),
        "masked_fraction": float(mask.mean().detach().cpu().item()),
    }
    return loss_total, stats


def _batch_covariance(x: torch.Tensor) -> torch.Tensor:
    """
    x: [batch, dim]
    """
    if x.shape[0] <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = x_centered.T @ x_centered / (x.shape[0] - 1)
    return cov


def mean_alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Align normalized batch means.
    """
    mu_a = F.normalize(z_a.mean(dim=0, keepdim=True), dim=1)
    mu_b = F.normalize(z_b.mean(dim=0, keepdim=True), dim=1)
    return F.mse_loss(mu_a, mu_b)


def coral_alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    CORAL-style covariance alignment.
    """
    cov_a = _batch_covariance(z_a)
    cov_b = _batch_covariance(z_b)
    d = z_a.shape[1]
    return ((cov_a - cov_b) ** 2).sum() / (4.0 * d * d)


def bridge_alignment_loss(
    z_bulk: torch.Tensor,
    z_pseudobulk: torch.Tensor,
    mean_weight: float = 1.0,
    coral_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_mean = mean_alignment_loss(z_bulk, z_pseudobulk)
    loss_coral = coral_alignment_loss(z_bulk, z_pseudobulk)
    loss_total = mean_weight * loss_mean + coral_weight * loss_coral

    stats = {
        "loss_mean": float(loss_mean.detach().cpu().item()),
        "loss_coral": float(loss_coral.detach().cpu().item()),
    }
    return loss_total, stats