from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_reflex.types import Decision, DecisionKind, DecisionRequest


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    kinds: frozenset[DecisionKind] = frozenset(DecisionKind)
    deterministic: bool = False
    local: bool = True
    probabilities: bool = False
    batching: bool = False
    trusted_for_high_risk: bool = False


class DecisionProvider(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> Decision | None:
        """Return a decision, or None when this provider cannot handle the request."""
        ...


def capabilities(provider: object) -> ProviderCapabilities:
    value = getattr(provider, "capabilities", None)
    return value if isinstance(value, ProviderCapabilities) else ProviderCapabilities()
