from __future__ import annotations

import json
import math
from time import perf_counter
from urllib import request as urlrequest
from urllib.parse import urlsplit

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.providers.jev_http import _NoRedirect
from agent_reflex.security import sanitize
from agent_reflex.types import Decision, DecisionRequest


class OpaPolicyProvider:
    """Optional decision provider backed by a local/remote OPA Data API endpoint.

    Expected OPA result can be a scalar choice/bool or an object with
    {"value": ..., "allow": bool, "reason": str}. Network failures return None
    via the engine's resilience path. For mandatory authorization use OpaPolicy
    in the engine's policies list; a cascade stage alone may fall through.
    """

    name = "opa"
    capabilities = ProviderCapabilities(deterministic=True, local=False, trusted_for_high_risk=True)

    def __init__(self, url: str = "http://127.0.0.1:8181/v1/data/agent_reflex/decision", timeout_s: float = 0.2):
        parsed = urlsplit(url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.scheme not in {"https", "http"}
        ):
            raise ValueError("invalid OPA URL")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("remote OPA endpoints require HTTPS")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("invalid OPA timeout")
        self.url = url
        self.timeout_s = timeout_s

    def decide(self, request: DecisionRequest) -> Decision | None:
        started = perf_counter()
        body = json.dumps(
            {
                "input": sanitize(
                    {
                        "state": request.state,
                        "question": request.question,
                        "kind": request.kind.value,
                        "options": list(request.options),
                        "risk": request.risk,
                        "trusted_context": request.trusted_context,
                        "untrusted_context": request.untrusted_context,
                        "metadata": request.metadata,
                    }
                )
            },
            default=str,
        ).encode()
        req = urlrequest.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.build_opener(_NoRedirect()).open(req, timeout=self.timeout_s) as resp:
            raw = resp.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("OPA response too large")
            payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        if result is None:
            return None
        reason = ""
        if isinstance(result, dict):
            if "value" in result:
                value = result["value"]
            elif "allow" in result:
                if type(result["allow"]) is not bool:
                    return None
                value = result["allow"]
            else:
                return None
            reason = str(result.get("reason", ""))
        else:
            value = result
        return Decision(
            value=value,
            confidence=1.0,
            provider=self.name,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={"policy_reason": reason, "opa_decision_id": payload.get("decision_id"), "deterministic": True},
        )
