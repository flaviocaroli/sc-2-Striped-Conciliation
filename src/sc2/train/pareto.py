from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class ParetoFront:
    minimize: tuple[str, ...]
    maximize: tuple[str, ...]
    points: list[dict[str, float]] = field(default_factory=list)

    def dominates(self, left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        weak = True
        strict = False
        for key in self.minimize:
            weak &= float(left[key]) <= float(right[key])
            strict |= float(left[key]) < float(right[key])
        for key in self.maximize:
            weak &= float(left[key]) >= float(right[key])
            strict |= float(left[key]) > float(right[key])
        return bool(weak and strict)

    def add(self, point: Mapping[str, float]) -> bool:
        candidate = {str(key): float(value) for key, value in point.items()}
        if any(self.dominates(existing, candidate) for existing in self.points):
            return False
        self.points = [existing for existing in self.points if not self.dominates(candidate, existing)]
        self.points.append(candidate)
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "minimize": list(self.minimize),
            "maximize": list(self.maximize),
            "points": self.points,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ParetoFront":
        output = cls(
            minimize=tuple(str(value) for value in state.get("minimize", [])),
            maximize=tuple(str(value) for value in state.get("maximize", [])),
        )
        output.points = [dict(point) for point in state.get("points", [])]  # type: ignore[arg-type]
        return output
