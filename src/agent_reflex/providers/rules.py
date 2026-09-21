from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionRequest

Predicate = Callable[[DecisionRequest], bool]
ValueFn = Callable[[DecisionRequest], Any] | Any


@dataclass(slots=True)
class Rule:
    name: str
    when: Predicate
    value: ValueFn
    confidence: float = 1.0


class RuleProvider:
    name = "rules"
    capabilities = ProviderCapabilities(deterministic=True, local=True, probabilities=False, trusted_for_high_risk=True)

    def __init__(self, rules: list[Rule] | None = None):
        self.rules = list(rules or [])

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def decide(self, request: DecisionRequest) -> Decision | None:
        started = perf_counter()
        for rule in self.rules:
            if rule.when(request):
                value = rule.value(request) if callable(rule.value) else rule.value
                return Decision(
                    value=value,
                    confidence=rule.confidence,
                    provider=self.name,
                    latency_ms=(perf_counter() - started) * 1000,
                    metadata={"rule": rule.name, "deterministic": True},
                )
        return None
