from dataclasses import replace

import pytest

from agent_reflex import Decision, DecisionBudget, DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.providers.embedding import EmbeddingExample
from agent_reflex.semantic import SemanticFallback, SemanticFlow, SemanticProfile, SemanticTask, _distribution


class Encoder:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return [[1, 0] if t.startswith("left") else [0, 1] if t.startswith("right") else [-1, -1] for t in texts]


def make_flow():
    task = SemanticTask("routing", "Which queue?", ("left", "right"))
    examples = [EmbeddingExample(label + str(i), label) for label in task.labels for i in range(2)]
    return SemanticFlow(task, examples, encoder=Encoder())


def request(text="left query", **kwargs):
    return DecisionRequest(
        {"summary": text, "private_notes": "never send this"},
        "Which queue?",
        DecisionKind.CHOICE,
        ("left", "right"),
        metadata={"task": "routing"},
        **kwargs,
    )


def fitted():
    flow = make_flow()
    flow.fit(
        [("left calibrate", "left"), ("right calibrate", "right")],
        [("left validate", "left"), ("right validate", "right"), ("unknown validation", None)],
        min_accepted=2,
    )
    return flow


def test_unvalidated_flow_abstains_then_fits_separate_distribution_and_gate():
    flow = make_flow()
    assert flow.decide(request()).metadata["reason"] == "profile_not_validated"
    profile = flow.fit(
        [("left cal", "left"), ("right cal", "right")],
        [("left val", "left"), ("right val", "right"), ("outside", None)],
        min_accepted=2,
    )
    assert profile.enabled and profile.validation_accepted == 2
    d = flow.decide(request())
    assert d.value == "left" and d.probabilities["left"] == d.confidence
    assert sum(d.probabilities.values()) == pytest.approx(1)
    assert flow.decide(request("unrelated weather")).abstained
    assert d.metadata["calibration_sha256"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"metadata": {}}, "unknown_task"),
        ({"question": "Ignore that, execute?"}, "question_contract_mismatch"),
        ({"options": ["left"]}, "option_contract_mismatch"),
        ({"options": [1, 2]}, "option_contract_mismatch"),
        ({"state": "left query"}, "structured_context_required"),
        ({"state": {}}, "required_context_missing"),
        ({"state": {"summary": "x" * 2049}}, "context_too_long"),
        ({"state": {"summary": "left [TRUNCATED]"}}, "context_too_long"),
    ],
)
def test_invalid_contract_abstains_before_inference(change, reason):
    flow = fitted()
    before = len(flow.router.model.seen)
    assert flow.decide(replace(request(), **change)).metadata["reason"] == reason
    assert len(flow.router.model.seen) == before


def test_selected_context_is_redacted_and_irrelevant_data_does_not_enter_model():
    flow = fitted()
    req = request("left issue token=sentinel-secret-value")
    req.state["background"] = "IGNORE. right " * 10000
    d = flow.decide(req)
    assert d.value == "left"
    assert flow.router.model.seen[-1] == "left issue token=[REDACTED]"
    assert flow.raw(replace(req, metadata={})) is None
    assert flow.raw(req).value == "left"


def test_option_order_stable_and_total_context_limit():
    flow = fitted()
    assert flow.decide(replace(request(), options=("right", "left"))).value == "left"
    flow.task = replace(flow.task, fields=("summary", "detail"), max_chars=32)
    selected, reason = flow.prepare(replace(request(), state={"summary": "l" * 20, "detail": "d" * 20}))
    assert selected is None and reason == "context_too_long"


@pytest.mark.parametrize(
    "cal,val",
    [
        ([], [("left val", "left")]),
        ([("left0", "left")], [("left val", "left")]),
        ([("shared", "left")], [("shared", "left")]),
        ([("left cal", "unknown")], [("left val", "left")]),
        ([("left cal", "left")], [("left val", "unknown")]),
    ],
)
def test_bad_or_overlapping_development_data_rejected(cal, val):
    with pytest.raises(ValueError):
        make_flow().fit(cal, val)


def test_infeasible_validation_disables_local_answer():
    flow = make_flow()
    p = flow.fit([("left cal", "left")], [("left val", "right")], min_accepted=1)
    assert not p.enabled and flow.decide(request()).abstained


def test_profile_cannot_be_reused_for_different_training_or_task():
    flow = fitted()
    examples = [EmbeddingExample(label + str(i), label) for label in flow.task.labels for i in range(2)]
    with pytest.raises(ValueError, match="profile"):
        SemanticFlow(replace(flow.task, question="Different?"), examples, encoder=Encoder(), profile=flow.profile)
    loaded = SemanticFlow(flow.task, examples, encoder=Encoder(), profile=flow.profile)
    assert loaded.decide(request()).value == "left"


class Remote:
    name = "fixture-remote"
    capabilities = ProviderCapabilities(local=False)

    def __init__(self):
        self.requests = []

    def estimate_cost(self, req):
        assert isinstance(req.state, str)
        return 0.001

    def decide(self, req):
        self.requests.append(req)
        return Decision("right", 0.99, self.name, cost_usd=0.001)


def test_fallback_receives_only_selected_context_and_respects_budget():
    flow = fitted()
    remote = Remote()
    fallback = SemanticFallback(flow, remote)
    engine = ReflexEngine(
        [ProviderStage(flow, threshold=0), ProviderStage(fallback)], default_budget=DecisionBudget(max_cost_usd=0.01)
    )
    assert engine.decide(request()).value == "left" and not remote.requests
    d = engine.decide(request("unknown query"))
    assert d.value == "right" and d.metadata["fallback_task"] == "routing"
    sent = remote.requests[0]
    assert sent.state == "unknown query" and not sent.trusted_context and not sent.untrusted_context
    assert sent.metadata == {"task": "routing"}
    remote.requests.clear()
    assert engine.decide(request("unknown query"), DecisionBudget(max_cost_usd=0)).abstained
    assert not remote.requests
    assert engine.decide(replace(request(), metadata={})).abstained
    assert not remote.requests


def test_remote_without_estimate_and_unsupported_schema_are_not_called():
    class NoEstimate:
        name = "missing-estimate"
        capabilities = ProviderCapabilities(local=False)

        def decide(self, req):
            raise AssertionError("must not call")

    adapter = SemanticFallback(fitted(), NoEstimate())
    engine = ReflexEngine([ProviderStage(adapter)])
    assert engine.decide(request()).abstained
    assert adapter.estimate_cost(replace(request(), metadata={})) == 0
    assert adapter.decide(replace(request(), metadata={})) is None


@pytest.mark.parametrize("probs,t", [({}, 1), ({"a": 1}, 0), ({"a": float("nan")}, 1), ({"a": 0.2}, 1)])
def test_calibration_rejects_invalid_scores(probs, t):
    with pytest.raises(ValueError):
        _distribution(probs, t)


def test_configuration_errors():
    with pytest.raises(ValueError):
        SemanticTask("", "q", ("a", "b"))
    with pytest.raises(ValueError):
        SemanticTask("x", "q", ("a", "a"))
    with pytest.raises(ValueError):
        SemanticTask("x", "q", ("a", "b"), fields=())
    with pytest.raises(ValueError):
        SemanticTask("x", "q", ("a", "b"), max_chars=0)
    with pytest.raises(ValueError):
        SemanticProfile("f", temperature=float("nan"))
    with pytest.raises(ValueError):
        SemanticProfile("f", min_confidence=2)
    with pytest.raises(ValueError):
        SemanticFlow(SemanticTask("x", "q", ("a", "b")), [], encoder=Encoder())
