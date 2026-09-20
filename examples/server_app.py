from agent_reflex import DecisionBudget, ProviderStage, ReflexEngine
from agent_reflex.providers import Prototype, PrototypeProvider, Rule, RuleProvider


def build_engine() -> ReflexEngine:
    rules=RuleProvider([
        Rule("deny-root-delete",lambda r:"rm -rf /" in str(r.state),"deny"),
        Rule("force-push-review",lambda r:"git push" in str(r.state) and "--force" in str(r.state),"review"),
        Rule("read-only-git",lambda r:str(r.state).startswith(("git status","git diff")),"allow"),
    ])
    router=PrototypeProvider([
        Prototype("read status inspect diff","allow"),
        Prototype("delete destructive irreversible","deny"),
        Prototype("force production deploy credential write","review"),
    ])
    return ReflexEngine([
        ProviderStage(rules,threshold=1.0,timeout_ms=25,sanitize_context=False),
        ProviderStage(router,threshold=0.75,max_risk=0.84,max_ood=0.75,timeout_ms=50),
    ],default_budget=DecisionBudget(deadline_ms=250,max_cost_usd=0,max_provider_calls=2))
