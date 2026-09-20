from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, Field

from agent_reflex.engine import DecisionBudget, ReflexEngine
from agent_reflex.types import DecisionKind, DecisionRequest


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int = 600):
        self.limit=max(1,int(requests_per_minute))
        self.events: dict[str, deque[float]]=defaultdict(deque)

    def allow(self, key: str) -> bool:
        now=time.monotonic(); q=self.events[key]; cutoff=now-60
        while q and q[0] < cutoff:
            q.popleft()
        if len(q)>=self.limit:
            return False
        q.append(now); return True



class DecideBody(BaseModel):
    state: Any
    question: str = Field(min_length=1, max_length=16_384)
    kind: DecisionKind = DecisionKind.BOOLEAN
    options: list[Any] = Field(default_factory=list, max_length=512)
    risk: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trusted_context: dict[str, Any] = Field(default_factory=dict)
    untrusted_context: dict[str, Any] = Field(default_factory=dict)
    deadline_ms: float | None = Field(default=None, gt=0, le=60_000)
    max_cost_usd: float | None = Field(default=None, ge=0, le=100)
    max_provider_calls: int | None = Field(default=None, ge=1, le=64)

def create_app(
    engine: ReflexEngine,
    *,
    api_keys: set[str] | None=None,
    requests_per_minute: int=600,
    max_body_bytes: int=1_048_576,
    allow_unauthenticated: bool=False,
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install agent-reflex[server]") from exc

    configured=api_keys if api_keys is not None else {x for x in os.getenv("AGENT_REFLEX_API_KEYS","").split(",") if x}
    if not configured and not allow_unauthenticated:
        raise RuntimeError("No API keys configured. Set AGENT_REFLEX_API_KEYS or explicitly allow unauthenticated mode.")
    limiter=SlidingWindowLimiter(requests_per_minute)

    app=FastAPI(title="Agent Reflex",version="1.0.0rc1",docs_url=None if not allow_unauthenticated else "/docs")

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        length=request.headers.get("content-length")
        if length and int(length)>max_body_bytes:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"request body too large"},status_code=413)
        return await call_next(request)

    async def authorize(x_api_key: str|None, x_tenant_id: str|None) -> str:
        if allow_unauthenticated and not configured:
            identity=x_tenant_id or "anonymous"
        else:
            if not x_api_key or x_api_key not in configured:
                raise HTTPException(status_code=401,detail="invalid API key")
            identity=f"{x_tenant_id or 'default'}:{x_api_key[-6:]}"
        if not limiter.allow(identity):
            raise HTTPException(status_code=429,detail="rate limit exceeded")
        return identity

    @app.get("/healthz")
    async def healthz():
        return {"status":"ok"}

    @app.get("/readyz")
    async def readyz():
        health=engine.health()
        if not health["ready"]:
            raise HTTPException(status_code=503,detail=health)
        return health

    @app.post("/v1/decide")
    async def decide(body: DecideBody, x_api_key: str|None=Header(default=None), x_tenant_id: str|None=Header(default=None)):
        identity=await authorize(x_api_key,x_tenant_id)
        req=DecisionRequest(state=body.state,question=body.question,kind=body.kind,options=body.options,
                            risk=body.risk,metadata={**body.metadata,"tenant":identity},trusted_context=body.trusted_context,
                            untrusted_context=body.untrusted_context)
        default=engine.default_budget
        budget=DecisionBudget(deadline_ms=body.deadline_ms or default.deadline_ms,
                              max_cost_usd=default.max_cost_usd if body.max_cost_usd is None else body.max_cost_usd,
                              max_provider_calls=body.max_provider_calls or default.max_provider_calls)
        d=await engine.adecide(req,budget)
        return {"value":d.value,"confidence":d.confidence,"provider":d.provider,"probabilities":d.probabilities,
                "latency_ms":d.latency_ms,"cost_usd":d.cost_usd,"abstained":d.abstained,"metadata":d.metadata,
                "request_id":req.request_id}

    return app
