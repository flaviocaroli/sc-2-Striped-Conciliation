from __future__ import annotations

import numpy as np


def corrupt_bulk_vector(
    x: np.ndarray,
    mask_prob: float = 0.15,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simple corruption for bulk denoising:
    - randomly zero a fraction of features
    - optionally add Gaussian noise

    Input and output are 1D float arrays.
    """
    rng = np.random.default_rng(seed)

    x_corrupt = x.copy()

    if mask_prob > 0:
        mask = rng.random(len(x_corrupt)) < mask_prob
        x_corrupt[mask] = 0.0

    if noise_std > 0:
        x_corrupt = x_corrupt + rng.normal(0.0, noise_std, size=len(x_corrupt)).astype(np.float32)
        x_corrupt = np.clip(x_corrupt, a_min=0.0, a_max=None)

    return x_corrupt.astype(np.float32)