from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MaskBatch:
    corrupted: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    observed_nonzero_mask: torch.Tensor


def make_random_gene_mask(
    x: torch.Tensor,
    mask_prob: float,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        return torch.rand(x.shape, generator=gen, device=x.device) < mask_prob
    return torch.rand_like(x, dtype=torch.float32) < mask_prob


def make_nonzero_dropout_mask(
    x: torch.Tensor,
    mask_prob: float,
    seed: int | None = None,
) -> torch.Tensor:
    random_mask = make_random_gene_mask(x, mask_prob=mask_prob, seed=seed)
    nonzero_mask = x > 0
    return random_mask & nonzero_mask


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> MaskBatch:
    corrupted = x.clone()
    corrupted[mask] = fill_value

    return MaskBatch(
        corrupted=corrupted,
        target=x.clone(),
        mask=mask,
        observed_nonzero_mask=x > 0,
    )


def make_fixed_numpy_masks(
    matrix: np.ndarray,
    mask_prob: float,
    seed: int,
    nonzero_only: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_mask = rng.random(matrix.shape) < mask_prob

    if nonzero_only:
        random_mask = random_mask & (matrix > 0)

    return random_mask


def save_mask_npz(path: str, mask: np.ndarray, seed: int, mask_prob: float) -> None:
    np.savez_compressed(
        path,
        mask=mask.astype(np.bool_),
        seed=np.array(seed),
        mask_prob=np.array(mask_prob),
    )


def load_mask_npz(path: str) -> np.ndarray:
    obj = np.load(path)
    return obj["mask"].astype(bool)