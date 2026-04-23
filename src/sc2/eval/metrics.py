from __future__ import annotations

import torch


def samplewise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Returns one MSE value per sample.
    Shape:
      pred   : [batch, features]
      target : [batch, features]
      output : [batch]
    """
    return ((pred - target) ** 2).mean(dim=1)


def samplewise_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Returns one MAE value per sample.
    Shape:
      pred   : [batch, features]
      target : [batch, features]
      output : [batch]
    """
    return (pred - target).abs().mean(dim=1)