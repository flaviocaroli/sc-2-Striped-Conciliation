from __future__ import annotations

import math
from typing import Mapping


def cosine_with_warmup(step: int, total_steps: int, *, warmup_fraction: float, end_ratio: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(step / float(total_steps), 0.0), 1.0)
    warmup = min(max(float(warmup_fraction), 0.0), 0.95)
    if warmup > 0.0 and progress < warmup:
        return max(progress / warmup, 1.0e-4)
    adjusted = (progress - warmup) / max(1.0 - warmup, 1.0e-8)
    return float(end_ratio) + (1.0 - float(end_ratio)) * 0.5 * (1.0 + math.cos(math.pi * adjusted))


def linear_ramp(step: int, *, start_step: int, end_step: int, start: float, end: float) -> float:
    if end_step <= start_step:
        return float(end)
    fraction = min(max((step - start_step) / float(end_step - start_step), 0.0), 1.0)
    return float(start) + fraction * (float(end) - float(start))


def scheduled_weights(base: Mapping[str, float], ramps: Mapping[str, Mapping[str, float]], step: int) -> dict[str, float]:
    values = {str(name): float(value) for name, value in base.items()}
    for name, spec in ramps.items():
        values[str(name)] = linear_ramp(
            step,
            start_step=int(spec["start_step"]),
            end_step=int(spec["end_step"]),
            start=float(spec["start"]),
            end=float(spec["end"]),
        )
    return values
