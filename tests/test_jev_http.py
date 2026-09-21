from __future__ import annotations

import json

import pytest

from agent_reflex import DecisionKind, DecisionRequest
from agent_reflex.providers import DirectJevProvider, JevHTTPError


def _transport(payload):
    def send(url, headers, body, timeout):
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert headers["Authorization"] == "Bearer secret"
        assert b"secret" not in body
        assert timeout == 3.0
        request = json.loads(body)
        assert request["model"] == "jev-1.13.0"
        return 200, {}, json.dumps(payload).encode()

    return send


def test_direct_jev_choice_contract_and_cost():
    provider = DirectJevProvider(
        api_key="secret",
        timeout_s=3.0,
        transport=_transport(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": "deny",
                        "probabilities": {"allow": 0.01, "review": 0.03, "deny": 0.96},
                        "confidence": 0.94,
                    }
                },
                "usage": {"input_tokens": 1000, "output_tokens": 0},
            }
        ),
    )
    decision = provider.decide(
        DecisionRequest(
            state="rm -rf /",
            question="How should this command be handled?",
            kind=DecisionKind.CHOICE,
            options=["allow", "review", "deny"],
        )
    )
    assert decision is not None
    assert decision.value == "deny"
    assert decision.confidence == pytest.approx(0.94)
    assert decision.cost_usd == pytest.approx(0.000042)
    assert decision.metadata["model"] == "jev-1.13.0"


def test_direct_jev_noul_derives_selected_class_confidence():
    provider = DirectJevProvider(
        api_key="secret",
        transport=_transport(
            {
                "model": "jev-1.13.0",
                "answers": {"answer": {"type": "noul", "noul": 0.18}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        ),
        timeout_s=3.0,
    )
    decision = provider.decide(
        DecisionRequest(
            state="Tests are failing",
            question="Is the task complete?",
            kind=DecisionKind.BOOLEAN,
        )
    )
    assert decision is not None
    assert decision.value is False
    assert decision.confidence == pytest.approx(0.82)
    assert decision.probabilities["true"] == pytest.approx(0.18)
    assert decision.metadata["confidence_source"] == "derived_from_noul"


def test_direct_jev_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider = DirectJevProvider(transport=lambda *_: (_ for _ in ()).throw(AssertionError("must not call")))
    with pytest.raises(JevHTTPError, match="TYPESAFE_API_KEY"):
        provider.decide(DecisionRequest(state="x", question="done?", kind=DecisionKind.BOOLEAN))


def test_direct_jev_rejects_insecure_remote_base_url():
    with pytest.raises(ValueError, match="HTTPS"):
        DirectJevProvider(base_url="http://example.com", api_key="secret")


def test_direct_jev_score_maps_expected_index_to_option():
    provider = DirectJevProvider(
        api_key="secret",
        transport=_transport(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "answer": {
                        "type": "score",
                        "score": 1.8,
                        "confidence": 0.88,
                        "probabilities": {"0": 0.03, "1": 0.14, "2": 0.83},
                    }
                },
                "usage": {"input_tokens": 20, "output_tokens": 0},
            }
        ),
        timeout_s=3.0,
    )
    decision = provider.decide(
        DecisionRequest(
            state="high severity",
            question="How severe?",
            kind=DecisionKind.SCORE,
            options=["low", "medium", "high"],
        )
    )
    assert decision is not None
    assert decision.value == "high"
    assert decision.confidence == pytest.approx(0.88)
