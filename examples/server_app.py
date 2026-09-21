from agent_reflex import DecisionBudget, ProviderStage, Reflex, ReflexEngine
from agent_reflex.providers import CheckKind, EvidenceCheck, Prototype, PrototypeProvider, Rule, RuleProvider


def build_engine() -> ReflexEngine:
    rules = RuleProvider(
        [
            Rule("deny-root-delete", lambda r: "rm -rf /" in str(r.state), "deny"),
            Rule("force-push-review", lambda r: "git push" in str(r.state) and "--force" in str(r.state), "review"),
            Rule(
                "read-only-git",
                lambda r: r.state in ("git status", "git status --short", "git diff", "git diff --stat"),
                "allow",
            ),
        ]
    )
    router = PrototypeProvider(
        [
            Prototype("read status inspect diff", "allow"),
            Prototype("delete destructive irreversible", "deny"),
            Prototype("force production deploy credential write", "review"),
        ]
    )
    engine = Reflex.local(
        checks=[
            EvidenceCheck(
                "Did this specific verification command pass? A nonzero exit status means the gate did not pass, even if some individual checks succeeded.",
                "exit_code",
                CheckKind.EXIT_ZERO,
            ),
            EvidenceCheck(
                "Did this probe return a successful HTTP 2xx response?", "status_code", CheckKind.HTTP_SUCCESS
            ),
        ]
    )
    engine.stages.extend(
        [
            ProviderStage(rules, threshold=1.0, timeout_ms=25, sanitize_context=False),
            ProviderStage(router, threshold=0.75, max_risk=0.84, max_ood=0.75, timeout_ms=50),
        ]
    )
    engine.default_budget = DecisionBudget(deadline_ms=250, max_cost_usd=0, max_provider_calls=3)
    return engine
