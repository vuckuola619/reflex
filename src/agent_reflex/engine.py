from __future__ import annotations

import asyncio
import inspect
import math
import threading
from dataclasses import dataclass, field, replace
from time import monotonic, perf_counter
from typing import Iterable

from agent_reflex.context import ContextSelector, SafeContextSelector
from agent_reflex.providers.base import DecisionProvider, capabilities
from agent_reflex.telemetry import DecisionEvent, NoopTelemetrySink, TelemetrySink
from agent_reflex.types import Attempt, Decision, DecisionKind, DecisionRequest


@dataclass(slots=True)
class DecisionBudget:
    deadline_ms: float = 1_000.0
    max_cost_usd: float = 0.01
    max_provider_calls: int = 4

    def validate(self) -> None:
        if self.deadline_ms <= 0 or self.max_cost_usd < 0 or self.max_provider_calls <= 0:
            raise ValueError("invalid decision budget")


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout_s = max(0.0, float(recovery_timeout_s))
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if monotonic() - self.opened_at >= self.recovery_timeout_s:
                self.opened_at = None
                self.failures = 0
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = monotonic()


@dataclass(slots=True)
class ProviderStage:
    provider: DecisionProvider
    threshold: float = 0.8
    max_risk: float = 1.0
    max_ood: float = 1.0
    timeout_ms: float = 500.0
    retries: int = 0
    sanitize_context: bool = True
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if not 0 <= self.max_risk <= 1:
            raise ValueError("max_risk must be in [0, 1]")
        if not 0 <= self.max_ood <= 1:
            raise ValueError("max_ood must be in [0, 1]")
        if self.timeout_ms <= 0 or self.retries < 0:
            raise ValueError("invalid stage timeout/retries")


def _valid_value(request: DecisionRequest, decision: Decision) -> tuple[bool, str]:
    if not isinstance(decision.provider, str) or not decision.provider:
        return False, "invalid_provider_name"
    for value, name in ((decision.confidence, "confidence"), (decision.latency_ms, "latency_ms"), (decision.cost_usd, "cost_usd")):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False, f"non_finite_{name}"
    if not 0 <= decision.confidence <= 1:
        return False, "confidence_out_of_range"
    if decision.latency_ms < 0 or decision.cost_usd < 0:
        return False, "negative_accounting"
    for k, p in decision.probabilities.items():
        if not isinstance(k, str) or not isinstance(p, (int, float)) or not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
            return False, "invalid_probability"
    if decision.abstained:
        return True, "provider_abstained"
    if request.kind == DecisionKind.BOOLEAN:
        return (isinstance(decision.value, bool), "invalid_boolean" if not isinstance(decision.value, bool) else "")
    if request.kind in {DecisionKind.CHOICE, DecisionKind.SCORE}:
        ok = decision.value in request.options
        return ok, "value_not_in_options" if not ok else ""
    return False, "unsupported_kind"


async def _invoke_provider(provider: object, request: DecisionRequest) -> Decision | None:
    adecide = getattr(provider, "adecide", None)
    if adecide is not None:
        result = adecide(request)
        return await result if inspect.isawaitable(result) else result
    decide = getattr(provider, "decide")
    if inspect.iscoroutinefunction(decide):
        return await decide(request)
    return await asyncio.to_thread(decide, request)


@dataclass(slots=True)
class ReflexEngine:
    stages: list[ProviderStage] = field(default_factory=list)
    abstain_on_uncertain: bool = True
    default_budget: DecisionBudget = field(default_factory=DecisionBudget)
    context_selector: ContextSelector = field(default_factory=SafeContextSelector)
    telemetry: TelemetrySink = field(default_factory=NoopTelemetrySink)
    fail_closed_risk: float = 0.85

    def __init__(
        self,
        stages: Iterable[ProviderStage] = (),
        abstain_on_uncertain: bool = True,
        *,
        default_budget: DecisionBudget | None = None,
        context_selector: ContextSelector | None = None,
        telemetry: TelemetrySink | None = None,
        fail_closed_risk: float = 0.85,
    ):
        self.stages = list(stages)
        self.abstain_on_uncertain = abstain_on_uncertain
        self.default_budget = default_budget or DecisionBudget()
        self.context_selector = context_selector or SafeContextSelector()
        self.telemetry = telemetry or NoopTelemetrySink()
        self.fail_closed_risk = fail_closed_risk

    async def adecide(self, request: DecisionRequest, budget: DecisionBudget | None = None) -> Decision:
        request.validate()
        budget = budget or self.default_budget
        budget.validate()
        started = perf_counter()
        deadline = monotonic() + budget.deadline_ms / 1000
        attempts: list[dict] = []
        best: Decision | None = None
        total_cost = 0.0
        provider_calls = 0
        terminal_reason = "all_stages_exhausted"

        for stage in self.stages:
            caps = capabilities(stage.provider)
            if request.kind not in caps.kinds:
                attempts.append({"provider": stage.provider.name, "skipped": "unsupported_kind"})
                continue
            if request.risk > stage.max_risk:
                attempts.append({"provider": stage.provider.name, "skipped": "risk"})
                continue
            if request.risk >= self.fail_closed_risk and not (caps.deterministic or caps.trusted_for_high_risk):
                attempts.append({"provider": stage.provider.name, "skipped": "high_risk_untrusted_provider"})
                continue
            if not stage.breaker.allow():
                attempts.append({"provider": stage.provider.name, "skipped": "circuit_open"})
                continue
            if provider_calls >= budget.max_provider_calls:
                terminal_reason = "provider_call_budget_exhausted"
                break
            estimate = getattr(stage.provider, "estimate_cost", None)
            if callable(estimate):
                try:
                    estimated = float(estimate(request))
                except Exception:
                    estimated = 0.0
                if total_cost + max(0.0, estimated) > budget.max_cost_usd:
                    attempts.append({"provider": stage.provider.name, "skipped": "cost_budget"})
                    continue

            remaining_ms = max(0.0, (deadline - monotonic()) * 1000)
            if remaining_ms <= 0:
                terminal_reason = "deadline_exhausted"
                break
            timeout_s = min(stage.timeout_ms, remaining_ms) / 1000
            provider_request = self.context_selector.select(request) if stage.sanitize_context and not caps.deterministic else request

            decision: Decision | None = None
            last_error = ""
            for retry_idx in range(stage.retries + 1):
                if provider_calls >= budget.max_provider_calls:
                    terminal_reason = "provider_call_budget_exhausted"
                    break
                provider_calls += 1
                try:
                    decision = await asyncio.wait_for(_invoke_provider(stage.provider, provider_request), timeout=timeout_s)
                    stage.breaker.success()
                    break
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    stage.breaker.failure()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = f"error:{type(exc).__name__}"
                    stage.breaker.failure()
                if retry_idx < stage.retries:
                    remaining_ms = max(0.0, (deadline - monotonic()) * 1000)
                    if remaining_ms <= 0:
                        terminal_reason = "deadline_exhausted"
                        break
                    await asyncio.sleep(min(0.05 * (2**retry_idx), remaining_ms / 1000))

            if decision is None:
                attempts.append({"provider": stage.provider.name, "skipped": last_error or "unsupported"})
                continue

            valid, reason = _valid_value(request, decision)
            if not valid:
                attempts.append({"provider": decision.provider, "skipped": "invalid_typed_output", "reason": reason})
                stage.breaker.failure()
                continue
            ood = float(decision.metadata.get("ood_score", 0.0) or 0.0)
            if not math.isfinite(ood) or not 0 <= ood <= 1:
                attempts.append({"provider": decision.provider, "skipped": "invalid_ood_score"})
                continue
            total_cost += decision.cost_usd
            if total_cost > budget.max_cost_usd:
                attempts.append({"provider": decision.provider, "skipped": "cost_budget_exceeded_after_call"})
                terminal_reason = "cost_budget_exhausted"
                break

            accepted = not decision.abstained and decision.confidence >= stage.threshold and ood <= stage.max_ood
            attempts.append({
                "provider": decision.provider,
                "confidence": decision.confidence,
                "threshold": stage.threshold,
                "ood_score": ood,
                "max_ood": stage.max_ood,
                "value": decision.value,
                "accepted": accepted,
                "latency_ms": decision.latency_ms,
                "cost_usd": decision.cost_usd,
            })
            if not decision.abstained and (best is None or decision.confidence > best.confidence):
                best = decision
            if accepted:
                decision.metadata = {**decision.metadata, "attempts": attempts, "total_cost_usd": total_cost, "provider_calls": provider_calls}
                self._emit(request, decision, started, len(attempts), "accepted")
                return decision

        if best is not None and not self.abstain_on_uncertain and request.risk < self.fail_closed_risk:
            best.metadata = {**best.metadata, "attempts": attempts, "forced": True, "total_cost_usd": total_cost, "provider_calls": provider_calls}
            self._emit(request, best, started, len(attempts), "forced_best")
            return best
        result = Decision(
            value=None,
            confidence=best.confidence if best else 0.0,
            provider="abstain",
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            cost_usd=total_cost,
            metadata={"attempts": attempts, "best_provider": best.provider if best else None, "reason": terminal_reason, "provider_calls": provider_calls},
        )
        self._emit(request, result, started, len(attempts), terminal_reason)
        return result

    def decide(self, request: DecisionRequest, budget: DecisionBudget | None = None) -> Decision:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.adecide(request, budget))
        raise RuntimeError("ReflexEngine.decide() cannot run inside an active event loop; use await adecide()")

    async def adecide_many(self, requests: Iterable[DecisionRequest], budget: DecisionBudget | None = None, *, concurrency: int = 32) -> list[Decision]:
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(req: DecisionRequest) -> Decision:
            async with semaphore:
                return await self.adecide(req, budget)

        return list(await asyncio.gather(*(one(r) for r in requests)))

    def probe_all(self, request: DecisionRequest) -> list[Attempt]:
        return asyncio.run(self.aprobe_all(request))

    async def aprobe_all(self, request: DecisionRequest) -> list[Attempt]:
        request.validate()
        out: list[Attempt] = []
        for stage in self.stages:
            if request.risk > stage.max_risk:
                continue
            try:
                d = await asyncio.wait_for(_invoke_provider(stage.provider, request), timeout=stage.timeout_ms / 1000)
            except Exception:
                continue
            if d is None:
                continue
            valid, reason = _valid_value(request, d)
            if not valid:
                out.append(Attempt(stage.provider.name, None, stage.threshold, False, reason=reason))
                continue
            out.append(Attempt(stage.provider.name, d, stage.threshold, d.confidence >= stage.threshold))
        return out

    def health(self) -> dict:
        stages=[]
        available=0
        for stage in self.stages:
            allowed=stage.breaker.allow()
            if allowed:
                available += 1
            stages.append({"provider":stage.provider.name,"circuit_available":allowed,"failures":stage.breaker.failures})
        return {"ready": bool(self.stages) and available>0, "stages": stages}

    def _emit(self, request: DecisionRequest, decision: Decision, started: float, attempts: int, reason: str) -> None:
        try:
            self.telemetry.emit(DecisionEvent(
                request_id=request.request_id,
                kind=request.kind.value,
                provider=decision.provider,
                value=decision.value,
                confidence=decision.confidence,
                abstained=decision.abstained,
                latency_ms=(perf_counter() - started) * 1000,
                cost_usd=decision.cost_usd,
                risk=request.risk,
                attempts=attempts,
                reason=reason,
                metadata={"request_metadata": request.metadata, "decision_metadata": decision.metadata},
            ))
        except Exception:
            # Telemetry must never break the decision path.
            pass
