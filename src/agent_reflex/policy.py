"""Operator-owned authorization, independent of probabilistic provider ordering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from agent_reflex.types import DecisionRequest


@dataclass(frozen=True)
class PolicyVerdict:
    allow: bool
    reason: str = "policy_denied"
    version: str = "unversioned"


class Policy(Protocol):
    def authorize(self, request: DecisionRequest, candidate: object = None) -> PolicyVerdict: ...


@dataclass(frozen=True)
class AllowlistPolicy:
    """Allow only exact action identifiers supplied by the trusted embedding harness."""

    actions: frozenset[str]
    version: str = "1"

    def authorize(self, request: DecisionRequest, candidate: object = None) -> PolicyVerdict:
        action = request.trusted_context.get("action")
        allowed = isinstance(action, str) and action in self.actions
        return PolicyVerdict(allowed, "allowed" if allowed else "action_not_authorized", self.version)


class OpaPolicy:
    """Mandatory OPA authorization. Undefined, malformed and unavailable deny.

    OPA must return literal boolean true (or an object with allow: true).
    This adapter runs before inference and again with the proposed candidate.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8181/v1/data/agent_reflex/allow",
        *,
        timeout_s: float = 0.2,
        version: str = "operator-configured",
    ):
        from agent_reflex.providers.opa import OpaPolicyProvider

        self.provider = OpaPolicyProvider(url, timeout_s)
        self.version = version

    def authorize(self, request: DecisionRequest, candidate: object = None) -> PolicyVerdict:
        check = replace(request, trusted_context={**request.trusted_context, "policy_candidate": candidate})
        try:
            decision = self.provider.decide(check)
        except Exception:
            return PolicyVerdict(False, "opa_unavailable", self.version)
        allowed = decision is not None and decision.value is True and not decision.abstained
        return PolicyVerdict(allowed, "allowed" if allowed else "opa_denied", self.version)
