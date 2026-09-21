import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from agent_reflex import Decision, ProviderStage, ReflexEngine
from agent_reflex.api import create_app
from agent_reflex.mcp import create_mcp_server
from agent_reflex.metrics import EvalRow, multiclass_brier
from agent_reflex.providers import DirectJevProvider, Rule, RuleProvider


def app(rpm=600, max_body=1048576):
    rules = RuleProvider([Rule("yes", lambda r: True, True)])
    return create_app(
        ReflexEngine([ProviderStage(rules)]), api_keys={"key"}, requests_per_minute=rpm, max_body_bytes=max_body
    )


def test_api_rejects_client_trusted_context():
    with TestClient(app()) as c:
        r = c.post(
            "/v1/decide",
            headers={"X-API-Key": "key"},
            json={"state": "x", "question": "q", "trusted_context": {"allow": True}},
        )
        assert r.status_code == 422


def test_rate_limit_cannot_be_bypassed_with_tenant_header():
    with TestClient(app(rpm=1)) as c:
        for i, status in enumerate([200, 429]):
            r = c.post(
                "/v1/decide", headers={"X-API-Key": "key", "X-Tenant-ID": str(i)}, json={"state": "x", "question": "q"}
            )
            assert r.status_code == status


def test_chunked_body_without_length_is_bounded():
    with TestClient(app(max_body=32)) as c:
        r = c.post("/v1/decide", content=iter([b"a" * 20, b"b" * 20]))
        assert r.status_code == 413


@pytest.mark.parametrize("length", ["invalid", "-1", "10000000"])
def test_invalid_or_excessive_content_length(length):
    with TestClient(app()) as c:
        r = c.post("/v1/decide", content=b"{}", headers={"Content-Length": length})
        assert r.status_code in [400, 413]


def test_validation_error_does_not_echo_secret():
    with TestClient(app()) as c:
        r = c.post("/v1/decide", json={"state": "x", "question": {"api_key": "sentinel-secret"}})
        assert r.status_code == 422
        assert "sentinel-secret" not in r.text


def test_readiness_when_no_providers():
    with TestClient(create_app(ReflexEngine(), api_keys={"key"})) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503


def test_empty_question_is_validation_error():
    with TestClient(app()) as c:
        assert (
            c.post("/v1/decide", headers={"X-API-Key": "key"}, json={"state": "x", "question": " "}).status_code == 422
        )


def test_boolean_brier_uses_wire_labels():
    rows = [EvalRow(True, Decision(True, 0.9, "p", probabilities={"true": 0.9, "false": 0.1}))]
    assert multiclass_brier(rows) == pytest.approx(0.02)


@pytest.mark.parametrize(
    "url", ["http://localhost.attacker.test", "http://127.0.0.1.attacker.test", "https://user:pass@api.typesafe.ai"]
)
def test_jev_endpoint_validation(url):
    with pytest.raises(ValueError):
        DirectJevProvider(base_url=url)


def test_mcp_official_sdk_contract():
    async def run():
        engine = ReflexEngine([ProviderStage(RuleProvider([Rule("allow", lambda r: True, "allow")]))])
        async with Client(create_mcp_server(engine)) as c:
            r = await c.call_tool("decide_choice", {"state": "x", "question": "q", "options": ["allow", "deny"]})
            assert not r.is_error
            content = r.structured_content or json.loads(r.content[0].text)
            assert content["value"] == "allow"

    asyncio.run(run())
