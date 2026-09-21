import io
import json

import pytest

from agent_reflex import DecisionRequest, Reflex
from agent_reflex.policy import AllowlistPolicy, OpaPolicy
from agent_reflex.providers.opa import OpaPolicyProvider


def opa(monkeypatch, payload):
    class Opener:
        def open(self, req, timeout):
            assert timeout == 0.2
            body = json.loads(req.data)
            assert body["input"]["state"] == {"value": 1}
            return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("agent_reflex.providers.opa.urlrequest.build_opener", lambda *args: Opener())


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"result": True}, True),
        ({"result": False}, False),
        ({"result": {"allow": True}}, True),
        ({"result": {"allow": False}}, False),
        ({"result": {"value": "review", "reason": "gate"}}, "review"),
        ({"result": {"allow": "false"}}, None),
        ({"result": {}}, None),
        ({}, None),
        ([], None),
    ],
)
def test_opa_preserves_strict_types_and_undefined(monkeypatch, payload, expected):
    opa(monkeypatch, payload)
    decision = OpaPolicyProvider().decide(DecisionRequest({"value": 1}, "q"))
    assert (decision.value if decision is not None else None) == expected


def test_opa_mandatory_denial_and_network_failure(monkeypatch):
    policy = OpaPolicy()
    request = DecisionRequest({"value": 1}, "q")
    opa(monkeypatch, {"result": {"allow": "yes"}})
    assert not policy.authorize(request).allow
    opa(monkeypatch, {"result": True})
    assert policy.authorize(request).allow
    monkeypatch.setattr(policy.provider, "decide", lambda _: (_ for _ in ()).throw(OSError("offline")))
    assert policy.authorize(request).reason == "opa_unavailable"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://example.com/opa", "https://user:secret@example.com"])
def test_opa_url_constraints(url):
    with pytest.raises(ValueError):
        OpaPolicyProvider(url)


def test_allowlist_uses_only_operator_context():
    policy = AllowlistPolicy(frozenset({"inspect"}))
    request = DecisionRequest({"exit_code": 0, "action": "inspect"}, "Did the command succeed?")
    engine = Reflex.local(policies=[policy])
    assert engine.decide(request).abstained
    request.trusted_context = {"action": "inspect"}
    assert engine.decide(request).value is True


def test_offline_default_rejects_missing_checks_and_unknown_questions():
    engine = Reflex.local()
    assert engine.decide(DecisionRequest({"checks": []}, "Did all required checks pass?")).abstained
    assert engine.decide(DecisionRequest({"checks": [True, True]}, "Did all required checks pass?")).value is True
    assert engine.decide(DecisionRequest({"status_code": 503}, "Did the HTTP request succeed?")).value is False
    assert engine.decide(DecisionRequest("Looks good", "Is this deployment safe?")).abstained
