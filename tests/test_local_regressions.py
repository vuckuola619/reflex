import asyncio

import pytest

from agent_reflex import Decision, DecisionBudget, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.metrics import EvalRow, negative_log_likelihood
from agent_reflex.providers import DirectJevProvider, RuleProvider


def test_jev_key_is_not_in_repr():
    assert "sentinel-credential" not in repr(DirectJevProvider(api_key="sentinel-credential"))


def test_jev_does_not_invent_observed_model():
    provider = DirectJevProvider(api_key="test", transport=lambda *_: (200, {}, b'{"answers":{"answer":{"noul":0.9}}}'))
    result = provider.decide(DecisionRequest("x", "ok?"))
    assert result.metadata["model"] == ""


def test_boolean_nll_accepts_wire_probability_labels():
    rows = [EvalRow(True, Decision(True, 0.9, "jev", probabilities={"true": 0.9, "false": 0.1}))]
    assert negative_log_likelihood(rows) == pytest.approx(0.10536051565782628)


def test_nll_missing_gold_class_is_zero_probability():
    assert negative_log_likelihood([EvalRow("b", Decision("a", 1, "p", probabilities={"a": 1}))]) > 20


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_budget_and_stage_rejected(value):
    with pytest.raises(ValueError):
        DecisionBudget(deadline_ms=value).validate()
    with pytest.raises(ValueError):
        DecisionBudget(max_cost_usd=value).validate()
    with pytest.raises(ValueError):
        ProviderStage(RuleProvider(), timeout_ms=value)


def test_retry_does_not_start_after_global_deadline():
    class Slow:
        name = "slow"
        calls = 0

        async def adecide(self, request):
            self.calls += 1
            await asyncio.sleep(0.5)

    p = Slow()
    engine = ReflexEngine([ProviderStage(p, timeout_ms=5, retries=3)])
    result = asyncio.run(engine.adecide(DecisionRequest("x", "ok?"), DecisionBudget(deadline_ms=20)))
    assert result.abstained
    assert p.calls == 1


@pytest.mark.parametrize("ood", ["bad", None, {}, float("nan")])
def test_bad_ood_abstains_instead_of_crashing(ood):
    class Bad:
        name = "bad"

        def decide(self, request):
            return Decision(True, 1, self.name, metadata={"ood_score": ood})

    assert ReflexEngine([ProviderStage(Bad())]).decide(DecisionRequest("x", "ok?")).abstained


def test_forced_best_cannot_bypass_ood():
    class Bad:
        name = "bad"

        def decide(self, request):
            return Decision(True, 1, self.name, metadata={"ood_score": 0.99})

    engine = ReflexEngine([ProviderStage(Bad(), max_ood=0.5)], abstain_on_uncertain=False)
    assert engine.decide(DecisionRequest("x", "ok?")).abstained
