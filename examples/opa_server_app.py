import os

from agent_reflex import DecisionBudget, ProviderStage, ReflexEngine
from agent_reflex.providers import OpaPolicyProvider, Prototype, PrototypeProvider


def build_engine() -> ReflexEngine:
    opa = OpaPolicyProvider(os.getenv("OPA_URL", "http://127.0.0.1:8181/v1/data/agent_reflex/decision"))
    router = PrototypeProvider([
        Prototype("read inspect status diff", "allow"),
        Prototype("delete destructive irreversible", "deny"),
        Prototype("deploy credential force push production", "review"),
    ])
    return ReflexEngine([
        ProviderStage(opa, threshold=1.0, timeout_ms=80, retries=1, sanitize_context=False),
        ProviderStage(router, threshold=0.80, max_risk=0.84, max_ood=0.70, timeout_ms=50),
    ], default_budget=DecisionBudget(deadline_ms=300, max_cost_usd=0, max_provider_calls=3))
