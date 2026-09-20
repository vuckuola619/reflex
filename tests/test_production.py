import asyncio
import json
from pathlib import Path

from agent_reflex import Decision, DecisionBudget, DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.calibration import TemperatureCalibrator
from agent_reflex.providers import Prototype, PrototypeProvider, Rule, RuleProvider
from agent_reflex.security import redact_text
from agent_reflex.telemetry import JsonlTelemetrySink


def test_deadline_timeout_abstains():
    class Slow:
        name = "slow"
        async def adecide(self, request):
            await asyncio.sleep(0.05)
            return Decision(True, 1.0, self.name)
    engine = ReflexEngine([ProviderStage(Slow(), threshold=0.5, timeout_ms=5)])
    d = asyncio.run(engine.adecide(DecisionRequest("x", "ok?"), DecisionBudget(deadline_ms=20, max_provider_calls=1)))
    assert d.abstained
    assert d.metadata["attempts"][0]["skipped"] == "timeout"


def test_circuit_breaker_opens_after_failures():
    class Broken:
        name = "broken"
        def decide(self, request):
            raise RuntimeError("boom")
    stage = ProviderStage(Broken(), timeout_ms=25)
    stage.breaker.failure_threshold = 2
    engine = ReflexEngine([stage])
    for _ in range(2):
        assert engine.decide(DecisionRequest("x", "ok?")).abstained
    d = engine.decide(DecisionRequest("x", "ok?"))
    assert d.metadata["attempts"][0]["skipped"] == "circuit_open"


def test_high_risk_skips_untrusted_probabilistic_provider():
    provider = PrototypeProvider([Prototype("delete production", "allow")])
    engine = ReflexEngine([ProviderStage(provider, threshold=0.0)], fail_closed_risk=0.8)
    d = engine.decide(DecisionRequest("delete production", "action?", DecisionKind.CHOICE, ["allow", "deny"], risk=0.95))
    assert d.abstained
    assert d.metadata["attempts"][0]["skipped"] == "high_risk_untrusted_provider"


def test_high_risk_allows_deterministic_rule():
    rules = RuleProvider([Rule("deny", lambda r: True, "deny")])
    engine = ReflexEngine([ProviderStage(rules, threshold=1.0)])
    d = engine.decide(DecisionRequest("delete", "action?", DecisionKind.CHOICE, ["allow", "deny"], risk=1.0))
    assert d.value == "deny"


def test_ood_gate_rejects_semantic_answer():
    class OOD:
        name = "ood"
        def decide(self, request):
            return Decision("a", 0.99, self.name, metadata={"ood_score": 0.95})
    engine = ReflexEngine([ProviderStage(OOD(), threshold=0.5, max_ood=0.5)])
    d = engine.decide(DecisionRequest("x", "pick", DecisionKind.CHOICE, ["a", "b"]))
    assert d.abstained
    assert d.metadata["attempts"][0]["accepted"] is False


def test_non_finite_confidence_is_firewalled():
    class Bad:
        name = "bad"
        def decide(self, request):
            return Decision("a", float("nan"), self.name)
    engine = ReflexEngine([ProviderStage(Bad(), threshold=0.0)])
    d = engine.decide(DecisionRequest("x", "pick", DecisionKind.CHOICE, ["a", "b"]))
    assert d.abstained
    assert d.metadata["attempts"][0]["reason"] == "non_finite_confidence"


def test_telemetry_redacts_secrets(tmp_path: Path):
    sink = JsonlTelemetrySink(tmp_path / "events.jsonl")
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    engine = ReflexEngine([ProviderStage(rules, threshold=1.0)], telemetry=sink)
    engine.decide(DecisionRequest("token=supersecretvalue", "safe?", metadata={"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"}))
    text = (tmp_path / "events.jsonl").read_text()
    assert "supersecretvalue" not in text
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "[REDACTED]" in text


def test_redactor_common_patterns():
    s = redact_text("Authorization: Bearer abcdefghijkl token=mysecrettoken AKIAABCDEFGHIJKLMNOP")
    assert "abcdefghijkl" not in s
    assert "mysecrettoken" not in s
    assert "AKIAABCDEFGHIJKLMNOP" not in s


def test_temperature_calibrator_fit_and_transform():
    cal = TemperatureCalibrator.fit([0.9, 0.8, 0.7, 0.6], [True, True, False, False])
    out = cal.transform(0.8)
    assert 0 < out < 1
    assert cal.temperature > 0


def test_async_batch_preserves_count():
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    engine = ReflexEngine([ProviderStage(rules, threshold=1.0)])
    reqs = [DecisionRequest(i, "ok?") for i in range(20)]
    out = asyncio.run(engine.adecide_many(reqs, concurrency=4))
    assert len(out) == 20
    assert all(x.value is True for x in out)


def test_health_reports_ready():
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    engine = ReflexEngine([ProviderStage(rules)])
    assert engine.health()["ready"] is True

def test_rest_api_auth_and_decision():
    from fastapi.testclient import TestClient
    from agent_reflex.api import create_app
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    app = create_app(ReflexEngine([ProviderStage(rules, threshold=1.0)]), api_keys={"test-key"}, requests_per_minute=10)
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/decide", json={"state":"x","question":"ok?"}).status_code == 401
    r = client.post("/v1/decide", headers={"X-API-Key":"test-key"}, json={"state":"x","question":"ok?"})
    assert r.status_code == 200
    assert r.json()["value"] is True
    assert r.json()["request_id"]


def test_rest_api_secure_by_default_without_keys():
    from agent_reflex.api import create_app
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    try:
        create_app(ReflexEngine([ProviderStage(rules)]), api_keys=set())
    except RuntimeError as exc:
        assert "No API keys configured" in str(exc)
    else:
        raise AssertionError("server must not start unauthenticated by default")
