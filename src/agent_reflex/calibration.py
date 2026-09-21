from __future__ import annotations

import math
from dataclasses import dataclass


def _clip(p: float) -> float:
    return min(1 - 1e-9, max(1e-9, float(p)))


@dataclass(slots=True)
class TemperatureCalibrator:
    """Calibrates confidence using temperature scaling on binary correctness labels.

    This treats the top-class confidence as a binary probability that the selected
    class is correct. Fit on a frozen validation split, never the final test split.
    """

    temperature: float = 1.0

    def transform(self, confidence: float) -> float:
        p = _clip(confidence)
        logit = math.log(p / (1 - p)) / max(self.temperature, 1e-6)
        return 1 / (1 + math.exp(-logit))

    @classmethod
    def fit(
        cls, confidences: list[float], correct: list[bool], grid: tuple[float, ...] | None = None
    ) -> TemperatureCalibrator:
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("confidences and correct must be non-empty and same length")
        grid = grid or tuple(x / 20 for x in range(2, 101))
        best_t, best_loss = 1.0, float("inf")
        for t in grid:
            cal = cls(t)
            loss = 0.0
            for p, y in zip(confidences, correct, strict=False):
                q = _clip(cal.transform(p))
                loss += -(math.log(q) if y else math.log(1 - q))
            if loss < best_loss:
                best_t, best_loss = t, loss
        return cls(best_t)
