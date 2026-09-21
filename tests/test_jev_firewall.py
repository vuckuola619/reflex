import io
import urllib.error

import pytest

from agent_reflex import DecisionKind, DecisionRequest
from agent_reflex.providers.jev_http import DirectJevProvider, JevHTTPError, _default_transport, _NoRedirect


@pytest.mark.parametrize("body", [b"bad json", b"[]", b"{}", b'{"answers":{"answer":{"noul":2}}}'])
def test_malformed_wire_response_rejected(body):
    provider = DirectJevProvider(api_key="sentinel", transport=lambda *_: (200, {}, body))
    with pytest.raises(JevHTTPError):
        provider.decide(DecisionRequest("x", "q"))


def test_http_status_error_does_not_expose_response_body():
    provider = DirectJevProvider(api_key="sentinel", transport=lambda *_: (401, {}, b"private diagnostic"))
    with pytest.raises(JevHTTPError, match="HTTP 401") as exc:
        provider.decide(DecisionRequest("x", "q"))
    assert "private" not in str(exc.value)


def test_default_transport_bounds_response_and_disables_redirect(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    class Opener:
        def open(self, req, timeout):
            return Response(b"x" * 1_048_577)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(JevHTTPError, match="size limit"):
        _default_transport("https://api.typesafe.ai/v1/systemone", {}, b"{}", 1)
    assert _NoRedirect().redirect_request(None, None, 302, "found", {}, "https://elsewhere.test") is None


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("sensitive-url"), urllib.error.HTTPError("sensitive-url", 503, "sensitive", {}, None)],
)
def test_transport_errors_are_redacted(monkeypatch, error):
    class Opener:
        def open(self, *args, **kwargs):
            raise error

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(JevHTTPError) as exc:
        _default_transport("https://api.typesafe.ai/v1/systemone", {}, b"{}", 1)
    assert "sensitive" not in str(exc.value)


def test_typed_numeric_choice_and_score_probability_labels():
    request = DecisionRequest("x", "q", DecisionKind.CHOICE, [1, 2])
    p = DirectJevProvider(api_key="x")
    d = p._convert(request, {"answer": {"choice": "2", "confidence": 0.9}}, 1)
    assert type(d.value) is int and d.value == 2
    request = DecisionRequest("x", "q", DecisionKind.SCORE, ["low", "high"])
    d = p._convert(request, {"answers": {"answer": {"score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}}}}, 1)
    assert d.value == "high" and d.probabilities == {"low": 0.2, "high": 0.8}
    for score in [-1, 2, float("nan")]:
        with pytest.raises(JevHTTPError, match="scale"):
            p._convert(request, {"answer": {"score": score}}, 1)


@pytest.mark.parametrize("url", ["https://api.typesafe.ai/v1", "https://api.typesafe.ai/v1/systemone"])
def test_jev_endpoint_normalization(url):
    assert DirectJevProvider(base_url=url).base_url == "https://api.typesafe.ai"
