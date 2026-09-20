from agent_reflex import DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.providers import Prototype, PrototypeProvider, Rule, RuleProvider

rules = RuleProvider([
    Rule("deny-destructive", lambda r: "rm -rf /" in str(r.state), "deny"),
])
semantic = PrototypeProvider([
    Prototype("read-only git status", "allow"),
    Prototype("destructive production delete", "deny"),
    Prototype("force push main branch", "review"),
])
engine = ReflexEngine([
    ProviderStage(rules, threshold=1.0),
    ProviderStage(semantic, threshold=0.70),
])

result = engine.decide(DecisionRequest(
    state="git status",
    question="How should this command be handled?",
    kind=DecisionKind.CHOICE,
    options=["allow", "review", "deny"],
    risk=0.1,
))
print(result)
