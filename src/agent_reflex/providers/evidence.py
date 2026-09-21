from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


class CheckKind(StrEnum):
    EXIT_ZERO = "exit_zero"
    HTTP_SUCCESS = "http_success"
    ALL_TRUE = "all_true"


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """Operator-owned mapping from an exact question to a structured predicate.

    A check interprets observations; it does not authenticate their source or
    authorize actions. Mandatory policies remain independent of these checks.
    """

    question: str
    field: str
    kind: CheckKind

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.field or not isinstance(self.kind, CheckKind):
            raise ValueError("check requires a question, field and CheckKind")


class EvidenceProvider:
    """Exact structured checks. Unsupported, absent or malformed evidence abstains.

    Free text (including claimed success and embedded instructions) never overrides
    the selected structured field. No string-to-number or boolean coercion occurs.
    """

    name = "evidence"
    capabilities = ProviderCapabilities(
        kinds=frozenset({DecisionKind.BOOLEAN}), deterministic=True, local=True, probabilities=True
    )

    def __init__(self, checks: Iterable[EvidenceCheck]):
        rows = list(checks)
        self.checks = {c.question: c for c in rows}
        if len(self.checks) != len(rows):
            raise ValueError("duplicate question contracts")

    def decide(self, request: DecisionRequest) -> Decision | None:
        started = perf_counter()
        check = self.checks.get(request.question)
        if request.kind != DecisionKind.BOOLEAN or check is None or not isinstance(request.state, Mapping):
            return None
        observed: Any = request.state.get(check.field)
        if check.kind == CheckKind.EXIT_ZERO:
            if type(observed) is not int:
                return None
            value = observed == 0
        elif check.kind == CheckKind.HTTP_SUCCESS:
            if type(observed) is not int or not 100 <= observed <= 599:
                return None
            value = 200 <= observed < 300
        else:
            if not isinstance(observed, list) or not observed or any(type(x) is not bool for x in observed):
                return None
            value = all(observed)
        return Decision(
            value,
            1.0,
            self.name,
            probabilities={"true": float(value), "false": float(not value)},
            latency_ms=(perf_counter() - started) * 1000,
            metadata={"check": check.kind.value, "field": check.field, "deterministic": True},
        )
