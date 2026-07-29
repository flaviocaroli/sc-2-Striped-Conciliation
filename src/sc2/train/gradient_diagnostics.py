from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def objective_gradient_diagnostics(
    components: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, float]:
    """Return norms and pairwise cosines without applying PCGrad."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    vectors: dict[str, torch.Tensor] = {}
    for name, component in components.items():
        gradients = torch.autograd.grad(
            component,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        vectors[name] = torch.cat(pieces).detach().float()
    output: dict[str, float] = {}
    names = sorted(vectors)
    for name in names:
        output[f"grad_norm/{name}"] = float(vectors[name].norm().item())
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            denominator = vectors[left].norm() * vectors[right].norm()
            cosine = (
                float(torch.dot(vectors[left], vectors[right]).item() / denominator.item())
                if denominator.item() > 0.0
                else float("nan")
            )
            output[f"grad_cosine/{left}__{right}"] = cosine
    return output
