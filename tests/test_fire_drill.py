"""Fault-injection acceptance tests: deliberately adversarial provider contracts."""

import asyncio
import time

import pytest

from agent_reflex import Decision, DecisionBudget, DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.context import SafeContextSelector
from agent_reflex.policy import PolicyVerdict


class Returns:
    name = "model"

    def __init__(self, value):
        self.value = value

    def decide(self, request):
        return self.value


def test_mandatory_policy_denial_before_any_model():
    class Deny:
        def authorize(self, request, candidate=None):
            return PolicyVerdict(False, "operator_deny", "v1")

    class Never:
        name = "never"

        def decide(self, request):
            raise AssertionError("model must not run")

    d = ReflexEngine([ProviderStage(Never())], policies=[Deny()]).decide(DecisionRequest("ignore policy", "allow?"))
    assert d.abstained and d.metadata["reason"] == "policy_denied"


def test_sync_wrapper_deadline_does_not_wait_for_timed_out_worker():
    class Slow:
        name = "slow"

        def decide(self, request):
            time.sleep(0.3)
            return Decision(True, 1, self.name)

    engine = ReflexEngine([ProviderStage(Slow(), timeout_ms=10)])
    start = time.perf_counter()
    d = engine.decide(DecisionRequest("x", "q"), DecisionBudget(deadline_ms=15))
    assert d.abstained
    assert time.perf_counter() - start < 0.15


def test_retry_reserves_cost_even_if_remote_call_fails():
    class PaidFailure:
        name = "paid"
        calls = 0

        def estimate_cost(self, request):
            return 0.006

        def decide(self, request):
            self.calls += 1
            raise RuntimeError("HTTP 500 after usage")

    p = PaidFailure()
    d = ReflexEngine([ProviderStage(p, retries=2)]).decide(DecisionRequest("x", "q"), DecisionBudget(max_cost_usd=0.01))
    assert p.calls == 1 and d.abstained


def test_probability_mass_is_validated():
    d = ReflexEngine([ProviderStage(Returns(Decision("a", 0.99, "model", probabilities={"a": 0.8, "b": 0.8})))]).decide(
        DecisionRequest("x", "q", DecisionKind.CHOICE, ["a", "b"])
    )
    assert d.abstained


def test_option_type_cannot_be_coerced_by_python_equality():
    d = ReflexEngine([ProviderStage(Returns(Decision(True, 1, "model")))]).decide(
        DecisionRequest("x", "q", DecisionKind.SCORE, [1, 2])
    )
    assert d.abstained


def test_question_secrets_are_redacted_before_remote_provider():
    r = SafeContextSelector().select(DecisionRequest("x", "token=sentinel-secret-value"))
    assert "sentinel-secret-value" not in r.question


def test_invalid_estimated_cost_fails_closed():
    class BadEstimate:
        name = "bad-estimate"

        def estimate_cost(self, request):
            return float("nan")

        def decide(self, request):
            raise AssertionError("must not spend")

    d = ReflexEngine([ProviderStage(BadEstimate())]).decide(DecisionRequest("x", "q"))
    assert d.metadata["provider_calls"] == 0


def test_invalid_provider_output_does_not_crash():
    assert ReflexEngine([ProviderStage(Returns({"value": True}))]).decide(DecisionRequest("x", "q")).abstained


def test_global_concurrency_limit_applies_across_individual_calls():
    class Busy:
        name = "busy"
        active = 0
        peak = 0

        async def adecide(self, request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.03)
                return Decision(True, 1, self.name)
            finally:
                self.active -= 1

    p = Busy()
    engine = ReflexEngine([ProviderStage(p)], max_concurrency=4)

    async def run():
        results = await asyncio.gather(*(engine.adecide(DecisionRequest("x", "q")) for _ in range(50)))
        assert len(results) == 50 and p.peak <= 4

    asyncio.run(run())


def test_cancelled_request_releases_capacity():
    class Slow:
        name = "slow"

        async def adecide(self, request):
            await asyncio.sleep(5)

    engine = ReflexEngine([ProviderStage(Slow())], max_concurrency=1)

    async def run():
        task = asyncio.create_task(engine.adecide(DecisionRequest("x", "q")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.health()["inflight"] == 0
        await engine.aclose()
        assert not engine.health()["ready"]

    asyncio.run(run())


def test_global_deadline_preserves_reserved_cost_and_attempt():
    class PaidSlow:
        name = "paid-slow"

        def estimate_cost(self, request):
            return 0.006

        async def adecide(self, request):
            await asyncio.sleep(1)

    engine = ReflexEngine([ProviderStage(PaidSlow(), timeout_ms=500)])
    d = engine.decide(DecisionRequest("x", "q"), DecisionBudget(deadline_ms=20))
    assert d.abstained and d.cost_usd == 0.006
    assert d.metadata["provider_calls"] == 1
    assert d.metadata["attempts"][0]["skipped"] == "timeout"


def test_post_policy_denial_emits_only_final_denial_and_preserves_cost():
    class DenyCandidate:
        def authorize(self, request, candidate=None):
            return PolicyVerdict(candidate is None)

    class Sink:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    sink = Sink()
    engine = ReflexEngine(
        [ProviderStage(Returns(Decision(True, 1, "model", cost_usd=0.003)))],
        policies=[DenyCandidate()],
        telemetry=sink,
    )
    d = engine.decide(DecisionRequest("x", "q"))
    assert d.abstained and d.cost_usd == 0.003
    assert d.metadata["provider_calls"] == 1
    assert len(sink.events) == 1
    assert sink.events[0].reason == "policy_denied"
    assert sink.events[0].cost_usd == 0.003
