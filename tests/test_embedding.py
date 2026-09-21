import math

import pytest

from agent_reflex import DecisionKind, DecisionRequest
from agent_reflex.providers.embedding import EmbeddingExample, EmbeddingRouterProvider


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, sentences, **kwargs):
        return [self.vectors[s] for s in sentences]


def router(query, **kwargs):
    return EmbeddingRouterProvider(
        [EmbeddingExample("positive", True), EmbeddingExample("negative", False)],
        encoder=Encoder({"positive": [1, 0], "negative": [0, 1], "q": query}),
        **kwargs,
    )


def test_local_semantic_classification_and_provenance():
    d = router([1, 0]).decide(DecisionRequest("q", "passed?"))
    assert d.value is True
    assert sum(d.probabilities.values()) == pytest.approx(1)
    assert d.metadata["revision"]
    assert d.metadata["confidence_kind"].startswith("uncalibrated")


def test_ood_and_ties_abstain_instead_of_false_certainty():
    assert router([-1, -1]).decide(DecisionRequest("q", "passed?")) is None
    assert router([1, 1]).decide(DecisionRequest("q", "passed?")) is None


def test_task_routing_and_typed_options():
    p = router([1, 0], task_id="build")
    assert p.decide(DecisionRequest("q", "passed?", metadata={"task": "payment"})) is None
    assert p.decide(DecisionRequest("q", "passed?", metadata={"task": "build"})).value is True
    assert router([1, 0]).decide(DecisionRequest("q", "q", DecisionKind.SCORE, [0, 1])) is None


@pytest.mark.parametrize("vector", [[0, 0], [math.nan, 1], [1]])
def test_invalid_encoder_output_rejected(vector):
    with pytest.raises(ValueError):
        router(vector).decide(DecisionRequest("q", "passed?"))


def test_custom_model_requires_revision_and_bad_configuration_rejected():
    with pytest.raises(ValueError, match="revision"):
        EmbeddingRouterProvider([EmbeddingExample("x", True)], model_name="custom")
    with pytest.raises(ValueError, match="temperature"):
        router([1, 0], temperature=float("nan"))
    with pytest.raises(ValueError, match="acceptance"):
        router([1, 0], min_margin=-1)
    with pytest.raises(ValueError, match="at least one"):
        EmbeddingRouterProvider([], encoder=Encoder({}))
