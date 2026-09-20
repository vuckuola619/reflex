import asyncio
import math
import time

import pytest

from agent_reflex import Decision, DecisionBudget, DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.calibration import TemperatureCalibrator
from agent_reflex.metrics import EvalRow, bootstrap_ci, negative_log_likelihood, paired_bootstrap_accuracy_delta, risk_coverage_curve, summarize
from agent_reflex.providers import CalibratedProvider, Prototype, PrototypeProvider, Rule, RuleProvider
from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.tuning import ProbeRow, tune_thresholds


def test_request_validation_edges():
    with pytest.raises(ValueError):
        DecisionRequest("x", "").validate()
    with pytest.raises(ValueError):
        DecisionRequest("x", "q", risk=float("nan")).validate()
    with pytest.raises(ValueError):
        DecisionRequest("x", "q", DecisionKind.CHOICE).validate()
    with pytest.raises(ValueError):
        DecisionRequest("x", "q", DecisionKind.CHOICE, ["a", "a"]).validate()
    with pytest.raises(ValueError):
        DecisionRequest("x", "q", DecisionKind.CHOICE, list(range(513))).validate()


def test_budget_and_stage_validation():
    with pytest.raises(ValueError):
        DecisionBudget(deadline_ms=0).validate()
    with pytest.raises(ValueError):
        ProviderStage(RuleProvider(), threshold=2)
    with pytest.raises(ValueError):
        ProviderStage(RuleProvider(), max_risk=-1)
    with pytest.raises(ValueError):
        ProviderStage(RuleProvider(), max_ood=2)
    with pytest.raises(ValueError):
        ProviderStage(RuleProvider(), timeout_ms=0)


def test_retry_then_success_and_budget_call_count():
    class Flaky:
        name = "flaky"
        def __init__(self): self.n = 0
        def decide(self, request):
            self.n += 1
            if self.n == 1: raise RuntimeError("temporary")
            return Decision(True, 0.9, self.name)
    p = Flaky()
    engine = ReflexEngine([ProviderStage(p, threshold=0.5, retries=1, timeout_ms=50)])
    d = engine.decide(DecisionRequest("x", "ok?"), DecisionBudget(max_provider_calls=2))
    assert d.value is True
    assert d.metadata["provider_calls"] == 2


def test_cost_estimate_skips_provider():
    class Costly:
        name = "costly"
        def estimate_cost(self, request): return 1.0
        def decide(self, request): raise AssertionError("must not be called")
    engine = ReflexEngine([ProviderStage(Costly())])
    d = engine.decide(DecisionRequest("x", "ok?"), DecisionBudget(max_cost_usd=0.01))
    assert d.abstained
    assert d.metadata["attempts"][0]["skipped"] == "cost_budget"


def test_cost_after_call_can_abort():
    class Costly:
        name = "costly"
        def decide(self, request): return Decision(True, 0.99, self.name, cost_usd=0.02)
    engine = ReflexEngine([ProviderStage(Costly(), threshold=0.5)])
    d = engine.decide(DecisionRequest("x", "ok?"), DecisionBudget(max_cost_usd=0.01))
    assert d.abstained
    assert d.metadata["reason"] == "cost_budget_exhausted"


def test_unsupported_kind_via_capabilities():
    class BoolOnly:
        name = "bool-only"
        capabilities = ProviderCapabilities(kinds=frozenset({DecisionKind.BOOLEAN}))
        def decide(self, request): return Decision(True, 1, self.name)
    engine = ReflexEngine([ProviderStage(BoolOnly())])
    d = engine.decide(DecisionRequest("x", "pick", DecisionKind.CHOICE, ["a", "b"]))
    assert d.abstained
    assert d.metadata["attempts"][0]["skipped"] == "unsupported_kind"


def test_forced_best_only_low_risk():
    p = PrototypeProvider([Prototype("weather rain", "search"), Prototype("python code", "shell")])
    engine = ReflexEngine([ProviderStage(p, threshold=0.999)], abstain_on_uncertain=False, fail_closed_risk=0.8)
    low = engine.decide(DecisionRequest("weather", "route", DecisionKind.CHOICE, ["search", "shell"], risk=0.2))
    assert not low.abstained and low.metadata["forced"]
    high = engine.decide(DecisionRequest("weather", "route", DecisionKind.CHOICE, ["search", "shell"], risk=0.9))
    assert high.abstained


def test_sync_decide_rejected_inside_event_loop():
    engine = ReflexEngine([ProviderStage(RuleProvider([Rule("x", lambda r: True, True)]))])
    async def inner():
        with pytest.raises(RuntimeError):
            engine.decide(DecisionRequest("x", "ok?"))
    asyncio.run(inner())


def test_probe_all_and_tuning():
    p = PrototypeProvider([Prototype("safe", True), Prototype("unsafe", False)], temperature=0.05)
    engine = ReflexEngine([ProviderStage(p, threshold=0.5)], abstain_on_uncertain=False)
    rows = []
    for state, gold in [("safe", True), ("unsafe", False)]:
        req = DecisionRequest(state, "classify")
        rows.append(ProbeRow(gold, engine.probe_all(req)))
    tuned = tune_thresholds(rows, ["prototype"], grid=(0.5, 0.9))
    assert tuned.coverage == 1.0
    assert tuned.accuracy == 1.0


def test_calibrated_provider_wraps_metadata():
    p = PrototypeProvider([Prototype("safe", True)], temperature=0.05)
    wrapped = CalibratedProvider(p, TemperatureCalibrator(temperature=2.0))
    d = wrapped.decide(DecisionRequest("safe", "ok?"))
    assert d is not None
    assert d.provider.endswith(":calibrated")
    assert "raw_confidence" in d.metadata


def test_metrics_extended_paths():
    rows = [
        EvalRow("a", Decision("a", 0.9, "p", probabilities={"a":0.9,"b":0.1}, latency_ms=1)),
        EvalRow("b", Decision("a", 0.6, "p", probabilities={"a":0.6,"b":0.4}, latency_ms=3)),
        EvalRow("a", Decision(None, 0.2, "abstain", abstained=True, latency_ms=2)),
    ]
    m = summarize(rows)
    assert m["latency_p99_ms"] == 3
    assert negative_log_likelihood(rows) > 0
    curve = risk_coverage_curve(rows, thresholds=(0.5, 0.95))
    assert curve[0]["coverage"] == pytest.approx(2/3)
    lo, hi = bootstrap_ci(rows, lambda xs: summarize(xs)["overall_accuracy"], iterations=100)
    assert 0 <= lo <= hi <= 1
    delta = paired_bootstrap_accuracy_delta(rows, rows, iterations=100)
    assert delta["mean"] == 0


def test_invalid_probability_is_firewalled():
    class BadProb:
        name = "badprob"
        def decide(self, request): return Decision("a", 0.9, self.name, probabilities={"a":1.2})
    d = ReflexEngine([ProviderStage(BadProb())]).decide(DecisionRequest("x","pick",DecisionKind.CHOICE,["a","b"]))
    assert d.abstained
    assert d.metadata["attempts"][0]["reason"] == "invalid_probability"


def test_telemetry_failure_never_breaks_decision():
    class BadSink:
        def emit(self, event): raise RuntimeError("telemetry down")
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    d = ReflexEngine([ProviderStage(rules)], telemetry=BadSink()).decide(DecisionRequest("x","ok?"))
    assert d.value is True
