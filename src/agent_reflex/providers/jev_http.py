from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, Mapping[str, str], bytes]]


class JevHTTPError(RuntimeError):
    """Raised when the TypeSafe System One HTTP API returns an unusable response."""


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, Mapping[str, str], bytes]:
    req = urllib.request.Request(url, data=body, method="POST", headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - fixed/validated base URL
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        # Never include request headers or the API key in the exception.
        response_body = exc.read()[:2048]
        raise JevHTTPError(f"TypeSafe HTTP {exc.code}: {response_body.decode('utf-8', 'replace')}") from None
    except urllib.error.URLError as exc:
        raise JevHTTPError(f"TypeSafe connection error: {exc.reason}") from None


@dataclass(slots=True)
class DirectJevProvider:
    """Dependency-free TypeSafe Jev provider using the documented System One HTTP contract.

    The API key is read from TYPESAFE_API_KEY by default and is never stored in Decision
    metadata, telemetry, or repr output. This provider is intended for reproducible paired
    benchmarking and production fallback without requiring the TypeSafe/Pydantic SDK stack.
    """

    model: str = "jev-1.13.0"
    api_key: str | None = None
    base_url: str = "https://api.typesafe.ai"
    timeout_s: float = 10.0
    input_price_per_billion_tokens_usd: float = 42.0
    transport: Transport = _default_transport

    name = "jev-http"
    capabilities = ProviderCapabilities(local=False, probabilities=True)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/v1/systemone"):
            self.base_url = self.base_url[: -len("/v1/systemone")]
        elif self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
        if not self.base_url.startswith("https://") and not self.base_url.startswith("http://127.0.0.1") and not self.base_url.startswith("http://localhost"):
            raise ValueError("TypeSafe base_url must use HTTPS except for localhost fixtures")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def _resolved_key(self) -> str:
        key = self.api_key or os.getenv("TYPESAFE_API_KEY", "")
        if not key.strip():
            raise JevHTTPError("TYPESAFE_API_KEY is not set")
        return key.strip()

    @staticmethod
    def _question(request: DecisionRequest) -> dict[str, Any]:
        if request.kind == DecisionKind.BOOLEAN:
            return {"type": "noul", "instructions": request.question}
        criteria = {str(option): None for option in request.options}
        if request.kind == DecisionKind.CHOICE:
            return {"type": "choice", "instructions": request.question, "criteria": criteria}
        if request.kind == DecisionKind.SCORE:
            # The wire API expects an ordered list for Score criteria.
            return {"type": "score", "instructions": request.question, "criteria": [str(option) for option in request.options]}
        raise ValueError(f"unsupported decision kind: {request.kind}")

    def _payload(self, request: DecisionRequest) -> bytes:
        return json.dumps(
            {
                "model": self.model,
                "state": request.state,
                "questions": {"answer": self._question(request)},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def estimate_cost(self, request: DecisionRequest) -> float:
        # Conservative byte-based estimate used only for pre-call budget routing. Actual
        # accounting is replaced with the API's input_tokens when available.
        approx_tokens = max(1, len(self._payload(request)) // 4)
        return approx_tokens * self.input_price_per_billion_tokens_usd / 1_000_000_000

    @staticmethod
    def _answer_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        answers = payload.get("answers")
        if isinstance(answers, Mapping) and isinstance(answers.get("answer"), Mapping):
            return answers["answer"]  # type: ignore[return-value]
        # Compatibility with early/community System One servers that returned the question
        # map directly instead of wrapping it under `answers`.
        direct = payload.get("answer")
        if isinstance(direct, Mapping):
            return direct
        raise JevHTTPError("TypeSafe response missing answers.answer")

    def _convert(self, request: DecisionRequest, payload: Mapping[str, Any], latency_ms: float) -> Decision:
        answer = self._answer_from_payload(payload)
        probabilities: dict[str, float] = {}
        confidence: float

        if request.kind == DecisionKind.BOOLEAN:
            p_true = float(answer.get("noul"))
            if not 0.0 <= p_true <= 1.0:
                raise JevHTTPError("invalid Noul probability")
            value = p_true >= 0.5
            confidence = p_true if value else 1.0 - p_true
            probabilities = {"false": 1.0 - p_true, "true": p_true}
            confidence_source = "derived_from_noul"
        elif request.kind == DecisionKind.CHOICE:
            value = answer.get("choice")
            raw_probabilities = answer.get("probabilities") or {}
            if isinstance(raw_probabilities, Mapping):
                probabilities = {str(k): float(v) for k, v in raw_probabilities.items()}
            confidence = float(answer.get("confidence", probabilities.get(str(value), 0.0)))
            confidence_source = "api"
        elif request.kind == DecisionKind.SCORE:
            raw_score = float(answer.get("score"))
            index = min(max(int(round(raw_score)), 0), len(request.options) - 1)
            value = request.options[index]
            raw_probabilities = answer.get("probabilities") or {}
            if isinstance(raw_probabilities, Mapping):
                probabilities = {str(k): float(v) for k, v in raw_probabilities.items()}
            confidence = float(answer.get("confidence", probabilities.get(str(index), 0.0)))
            confidence_source = "api"
        else:  # pragma: no cover - DecisionRequest validation constrains this
            raise JevHTTPError("unsupported decision kind")

        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, Mapping) else 0
        output_tokens = int(usage.get("output_tokens", 0)) if isinstance(usage, Mapping) else 0
        cost = input_tokens * self.input_price_per_billion_tokens_usd / 1_000_000_000
        model = str(payload.get("model") or self.model)

        return Decision(
            value=value,
            confidence=confidence,
            provider=self.name,
            probabilities=probabilities,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "confidence_source": confidence_source,
                "transport": "typesafe-systemone-http",
            },
        )

    def decide(self, request: DecisionRequest) -> Decision | None:
        request.validate()
        started = perf_counter()
        key = self._resolved_key()
        body = self._payload(request)
        status, _, raw = self.transport(
            f"{self.base_url}/v1/systemone",
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "agent-reflex/1.0.0rc2",
            },
            body,
            self.timeout_s,
        )
        if status < 200 or status >= 300:
            raise JevHTTPError(f"TypeSafe HTTP {status}")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise JevHTTPError("TypeSafe returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise JevHTTPError("TypeSafe returned a non-object response")
        return self._convert(request, payload, (perf_counter() - started) * 1000)

    async def adecide(self, request: DecisionRequest) -> Decision | None:
        return await asyncio.to_thread(self.decide, request)
