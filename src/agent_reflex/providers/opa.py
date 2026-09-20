from __future__ import annotations

import json
from time import perf_counter
from urllib import request as urlrequest

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.security import sanitize
from agent_reflex.types import Decision, DecisionRequest


class OpaPolicyProvider:
    """Hard-policy provider backed by a local/remote OPA Data API endpoint.

    Expected OPA result can be a scalar choice/bool or an object with
    {"value": ..., "allow": bool, "reason": str}. Network failures return None
    via the engine's resilience path; policy deployments should normally run OPA
    as a local sidecar and configure high-risk decisions to fail closed/abstain.
    """
    name="opa"
    capabilities=ProviderCapabilities(deterministic=True, local=False, trusted_for_high_risk=True)

    def __init__(self, url: str="http://127.0.0.1:8181/v1/data/agent_reflex/decision", timeout_s: float=0.2):
        self.url=url
        self.timeout_s=timeout_s

    def decide(self, request: DecisionRequest) -> Decision | None:
        started=perf_counter()
        body=json.dumps({"input": sanitize({
            "state":request.state,
            "question":request.question,
            "kind":request.kind.value,
            "options":list(request.options),
            "risk":request.risk,
            "trusted_context":request.trusted_context,
            "untrusted_context":request.untrusted_context,
            "metadata":request.metadata,
        })},default=str).encode()
        req=urlrequest.Request(self.url,data=body,headers={"Content-Type":"application/json"},method="POST")
        with urlrequest.urlopen(req,timeout=self.timeout_s) as resp:
            payload=json.loads(resp.read())
        result=payload.get("result")
        if result is None:
            return None
        reason=""
        if isinstance(result,dict):
            if "value" in result:
                value=result["value"]
            elif "allow" in result:
                value=bool(result["allow"])
            else:
                return None
            reason=str(result.get("reason",""))
        else:
            value=result
        return Decision(value=value, confidence=1.0, provider=self.name,
                        latency_ms=(perf_counter()-started)*1000,
                        metadata={"policy_reason":reason,"opa_decision_id":payload.get("decision_id"),"deterministic":True})
