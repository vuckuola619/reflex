"""An offline default for verifiable outcomes, with explicit question contracts."""

from collections.abc import Iterable

from agent_reflex.engine import DecisionBudget, ProviderStage, ReflexEngine
from agent_reflex.policy import Policy
from agent_reflex.providers.evidence import CheckKind, EvidenceCheck, EvidenceProvider

COMMAND_PASSED = "Did the command succeed?"
HTTP_SUCCEEDED = "Did the HTTP request succeed?"
CHECKS_PASSED = "Did all required checks pass?"


class Reflex(ReflexEngine):
    @classmethod
    def local(cls, *, checks: Iterable[EvidenceCheck] = (), policies: Iterable[Policy] = ()) -> "Reflex":
        """No model download, credentials or network access.

        Supply your own exact question contracts via checks. This configuration
        abstains on semantic questions; append an explicitly configured provider
        when your deployment needs them. State is evidence, not authorization.
        """
        provider = EvidenceProvider(
            [
                EvidenceCheck(COMMAND_PASSED, "exit_code", CheckKind.EXIT_ZERO),
                EvidenceCheck(HTTP_SUCCEEDED, "status_code", CheckKind.HTTP_SUCCESS),
                EvidenceCheck(CHECKS_PASSED, "checks", CheckKind.ALL_TRUE),
                *checks,
            ]
        )
        return cls(
            [ProviderStage(provider, threshold=1)],
            policies=policies,
            default_budget=DecisionBudget(deadline_ms=250, max_cost_usd=0, max_provider_calls=4),
        )
