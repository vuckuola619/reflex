import asyncio
from itertools import repeat

import pytest

from agent_reflex import Decision, DecisionBudget, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.engine import CircuitBreaker


def test_health_does_not_consume_single_half_open_probe():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0)
    breaker.failure()
    assert breaker.available() and breaker.available()
    assert breaker.allow()
    assert not breaker.allow() and not breaker.available()
    breaker.success()
    assert breaker.available() and breaker.allow()


def test_half_open_provider_recovery_and_reopen():
    class Flaky:
        name = "flaky"
        failing = True

        def decide(self, request):
            if self.failing:
                raise OSError("offline")
            return Decision(True, 1, self.name)

    provider = Flaky()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0)
    engine = ReflexEngine([ProviderStage(provider, breaker=breaker)])
    request = DecisionRequest("x", "q")
    assert engine.decide(request).abstained
    assert engine.health()["ready"]
    assert engine.health()["ready"]
    provider.failing = False
    assert engine.decide(request).value is True
    provider.failing = True
    assert engine.decide(request).abstained


def test_batch_caps_input_and_live_task_count_preserving_order():
    class Echo:
        name = "echo"
        peak_tasks = 0

        async def adecide(self, request):
            self.peak_tasks = max(self.peak_tasks, len(asyncio.all_tasks()))
            await asyncio.sleep(0)
            return Decision(request.state % 2 == 0, 1, self.name)

    provider = Echo()
    engine = ReflexEngine([ProviderStage(provider)])

    async def run():
        rows = (DecisionRequest(i, "even?") for i in range(200))
        results = await engine.adecide_many(rows, concurrency=4)
        assert [d.value for d in results] == [i % 2 == 0 for i in range(200)]
        assert provider.peak_tasks <= 12
        with pytest.raises(ValueError, match="max_items"):
            await engine.adecide_many(repeat(DecisionRequest(1, "q")), max_items=8)
        assert engine.health()["inflight"] == 0
        await engine.aclose()

    asyncio.run(run())


def test_invalid_billed_response_retains_cost_and_blocks_further_spend():
    class Invalid:
        name = "invalid"

        def decide(self, request):
            return Decision("wrong type", 1, self.name, cost_usd=0.02)

    class Never:
        name = "never"

        def decide(self, request):
            raise AssertionError("budget already spent")

    d = ReflexEngine([ProviderStage(Invalid()), ProviderStage(Never())]).decide(
        DecisionRequest("x", "q"), DecisionBudget(max_cost_usd=0.01)
    )
    assert d.abstained and d.cost_usd == 0.02
    assert d.metadata["provider_calls"] == 1
