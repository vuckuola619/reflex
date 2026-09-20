from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from agent_reflex.types import Attempt


@dataclass(slots=True)
class ProbeRow:
    gold: Any
    attempts: list[Attempt]


@dataclass(slots=True)
class TuneResult:
    thresholds: dict[str, float]
    score: float
    accuracy: float
    coverage: float
    avg_cost_usd: float
    avg_latency_ms: float


def tune_thresholds(
    rows: list[ProbeRow],
    provider_order: list[str],
    grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    error_penalty: float = 3.0,
    abstain_penalty: float = 0.25,
    cost_weight: float = 10.0,
    latency_weight: float = 0.0005,
) -> TuneResult:
    """Grid-search cascade thresholds on held-out data.

    Score = correct - penalties - cost - latency. Tune weights for your workload.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    best: TuneResult | None = None
    by_name_rows = [({a.provider: a for a in row.attempts}, row.gold) for row in rows]

    for values in product(grid, repeat=len(provider_order)):
        thresholds = dict(zip(provider_order, values))
        correct = 0
        answered = 0
        total_cost = 0.0
        total_latency = 0.0
        utility = 0.0
        for attempts, gold in by_name_rows:
            picked = None
            for name in provider_order:
                attempt = attempts.get(name)
                if attempt is None:
                    continue
                total_latency += attempt.decision.latency_ms
                total_cost += attempt.decision.cost_usd
                if attempt.decision.confidence >= thresholds[name]:
                    picked = attempt.decision
                    break
            if picked is None:
                utility -= abstain_penalty
                continue
            answered += 1
            if picked.value == gold:
                correct += 1
                utility += 1.0
            else:
                utility -= error_penalty
        n = len(rows)
        utility -= total_cost * cost_weight
        utility -= total_latency * latency_weight
        result = TuneResult(
            thresholds=thresholds,
            score=utility / n,
            accuracy=correct / n,
            coverage=answered / n,
            avg_cost_usd=total_cost / n,
            avg_latency_ms=total_latency / n,
        )
        if best is None or result.score > best.score:
            best = result
    assert best is not None
    return best
