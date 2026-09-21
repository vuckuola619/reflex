from agent_reflex import DecisionKind, DecisionRequest, EvalRow, ProviderStage, ReflexEngine, summarize
from agent_reflex.providers import Prototype, PrototypeProvider, Rule, RuleProvider


def test_rules_win_before_semantic():
    rules = RuleProvider([Rule("block", lambda r: "rm -rf /" in str(r.state), "deny")])
    proto = PrototypeProvider([Prototype("delete files", "allow")])
    engine = ReflexEngine([ProviderStage(rules, 1.0), ProviderStage(proto, 0.1)])
    d = engine.decide(DecisionRequest("rm -rf /", "handle?", DecisionKind.CHOICE, ["allow", "deny"]))
    assert d.value == "deny"
    assert d.provider == "rules"


def test_abstains_when_uncertain():
    proto = PrototypeProvider([Prototype("weather", "search"), Prototype("code", "shell")])
    engine = ReflexEngine([ProviderStage(proto, 0.99)])
    d = engine.decide(DecisionRequest("something unrelated", "route?", DecisionKind.CHOICE, ["search", "shell"]))
    assert d.abstained


def test_summary_metrics():
    proto = PrototypeProvider([Prototype("safe", True), Prototype("unsafe", False)], temperature=0.05)
    engine = ReflexEngine([ProviderStage(proto, 0.5)], abstain_on_uncertain=False)
    rows = []
    for state, gold in [("safe", True), ("unsafe", False)]:
        d = engine.decide(DecisionRequest(state, "classify state", DecisionKind.BOOLEAN))
        rows.append(EvalRow(gold, d))
    m = summarize(rows)
    assert m["overall_accuracy"] == 1.0
    assert m["coverage"] == 1.0


def test_invalid_provider_output_never_escapes():
    class BadProvider:
        name = "bad"

        def decide(self, request):
            from agent_reflex import Decision

            return Decision(value="NOT_AN_OPTION", confidence=1.0, provider=self.name)

    engine = ReflexEngine([ProviderStage(BadProvider(), 0.0)])
    d = engine.decide(DecisionRequest("x", "pick", DecisionKind.CHOICE, ["a", "b"]))
    assert d.abstained
    assert d.value is None
    assert d.metadata["attempts"][0]["skipped"] == "invalid_typed_output"
