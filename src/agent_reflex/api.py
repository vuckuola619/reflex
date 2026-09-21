from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response
    from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel, ConfigDict, Field

from agent_reflex.engine import DecisionBudget, ReflexEngine
from agent_reflex.types import DecisionKind, DecisionRequest


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int = 600):
        self.limit = max(1, int(requests_per_minute))
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.events[key]
        cutoff = now - 60
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


class DecideBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    state: Any
    question: str = Field(min_length=1, max_length=16_384)
    kind: DecisionKind = DecisionKind.BOOLEAN
    options: list[Any] = Field(default_factory=list, max_length=512)
    risk: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    untrusted_context: dict[str, Any] = Field(default_factory=dict)
    deadline_ms: float | None = Field(default=None, gt=0, le=60_000)
    max_cost_usd: float | None = Field(default=None, ge=0, le=100)
    max_provider_calls: int | None = Field(default=None, ge=1, le=64)


def create_app(
    engine: ReflexEngine,
    *,
    api_keys: set[str] | None = None,
    requests_per_minute: int = 600,
    max_body_bytes: int = 1_048_576,
    allow_unauthenticated: bool = False,
    trusted_context: dict[str, Any] | None = None,
) -> FastAPI:
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install agent-reflex[server]") from exc

    configured = (
        api_keys if api_keys is not None else {x for x in os.getenv("AGENT_REFLEX_API_KEYS", "").split(",") if x}
    )
    if not configured and not allow_unauthenticated:
        raise RuntimeError(
            "No API keys configured. Set AGENT_REFLEX_API_KEYS or explicitly allow unauthenticated mode."
        )
    limiter = SlidingWindowLimiter(requests_per_minute)

    from agent_reflex import __version__

    app = FastAPI(title="Agent Reflex", version=__version__, docs_url=None if not allow_unauthenticated else "/docs")

    @app.middleware("http")
    async def body_limit(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        length = request.headers.get("content-length")
        if length:
            try:
                declared = int(length)
            except ValueError:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
            if declared < 0:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
            if declared > max_body_bytes:
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        if request.method == "POST":
            size = 0
            chunks = []
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > max_body_bytes:
                            return JSONResponse({"detail": "request body too large"}, status_code=413)
                        chunks.append(chunk)
            except TimeoutError:
                return JSONResponse({"detail": "request body timeout"}, status_code=408)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        return JSONResponse({"detail": "invalid request"}, status_code=422)

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> Response:
        return JSONResponse({"detail": "invalid request"}, status_code=422)

    async def authorize(x_api_key: str | None, x_tenant_id: str | None) -> str:
        if allow_unauthenticated and not configured:
            identity = "anonymous"
        else:
            if not x_api_key or not any(hmac.compare_digest(x_api_key.encode(), key.encode()) for key in configured):
                raise HTTPException(status_code=401, detail="invalid API key")
            identity = hashlib.sha256(x_api_key.encode()).hexdigest()
        if not limiter.allow(identity):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return identity

    @app.get("/healthz")
    @app.get("/health")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    @app.get("/ready")
    async def readyz() -> dict[str, Any]:
        health = engine.health()
        if not health["ready"]:
            raise HTTPException(status_code=503, detail=health)
        return health

    @app.post("/v1/decide")
    async def decide(
        body: DecideBody, x_api_key: str | None = Header(default=None), x_tenant_id: str | None = Header(default=None)
    ) -> dict[str, Any]:
        identity = await authorize(x_api_key, x_tenant_id)
        req = DecisionRequest(
            state=body.state,
            question=body.question,
            kind=body.kind,
            options=body.options,
            risk=body.risk,
            metadata={**body.metadata, "tenant": identity},
            trusted_context=dict(trusted_context or {}),
            untrusted_context=body.untrusted_context,
        )
        default = engine.default_budget
        budget = DecisionBudget(
            deadline_ms=min(body.deadline_ms or default.deadline_ms, default.deadline_ms),
            max_cost_usd=default.max_cost_usd
            if body.max_cost_usd is None
            else min(body.max_cost_usd, default.max_cost_usd),
            max_provider_calls=min(body.max_provider_calls or default.max_provider_calls, default.max_provider_calls),
        )
        d = await engine.adecide(req, budget)
        return {
            "value": d.value,
            "confidence": d.confidence,
            "provider": d.provider,
            "probabilities": d.probabilities,
            "latency_ms": d.latency_ms,
            "cost_usd": d.cost_usd,
            "abstained": d.abstained,
            "metadata": d.metadata,
            "request_id": req.request_id,
        }

    return app
