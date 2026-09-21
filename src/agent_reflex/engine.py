from __future__ import annotations

import asyncio
import inspect
import math
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from time import monotonic, perf_counter
from typing import Any, cast

from agent_reflex.context import ContextSelector, SafeContextSelector
from agent_reflex.policy import Policy, PolicyVerdict
from agent_reflex.providers.base import DecisionProvider, capabilities
from agent_reflex.telemetry import DecisionEvent, NoopTelemetrySink, TelemetrySink
from agent_reflex.types import Attempt, Decision, DecisionKind, DecisionRequest


@dataclass(slots=True)
class DecisionBudget:
    deadline_ms: float = 1_000.0
    max_cost_usd: float = 0.01
    max_provider_calls: int = 4

    def validate(self) -> None:
        if (
            not math.isfinite(self.deadline_ms)
            or not math.isfinite(self.max_cost_usd)
            or self.deadline_ms <= 0
            or self.max_cost_usd < 0
            or type(self.max_provider_calls) is not int
            or self.max_provider_calls <= 0
        ):
            raise ValueError("invalid decision budget")


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout_s = max(0.0, float(recovery_timeout_s))
        self.failures = 0
        self.opened_at: float | None = None
        self._probing = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if not self._probing and monotonic() - self.opened_at >= self.recovery_timeout_s:
                self._probing = True
                return True
            return False

    def available(self) -> bool:
        """Read-only readiness; observing health must never consume a recovery probe."""
        with self._lock:
            return self.opened_at is None or (
                not self._probing and monotonic() - self.opened_at >= self.recovery_timeout_s
            )

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None
            self._probing = False

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            self._probing = False
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
        if (
            not math.isfinite(self.timeout_ms)
            or self.timeout_ms <= 0
            or type(self.retries) is not int
            or not 0 <= self.retries <= 10
        ):
            raise ValueError("invalid stage timeout/retries")


def _valid_value(request: DecisionRequest, decision: Decision) -> tuple[bool, str]:
    if not isinstance(decision, Decision):
        return False, "invalid_decision_type"
    if (
        not isinstance(decision.probabilities, dict)
        or not isinstance(decision.metadata, dict)
        or type(decision.abstained) is not bool
    ):
        return False, "invalid_decision_fields"
    if not isinstance(decision.provider, str) or not decision.provider:
        return False, "invalid_provider_name"
    for value, name in (
        (decision.confidence, "confidence"),
        (decision.latency_ms, "latency_ms"),
        (decision.cost_usd, "cost_usd"),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False, f"non_finite_{name}"
    if not 0 <= decision.confidence <= 1:
        return False, "confidence_out_of_range"
    if decision.latency_ms < 0 or decision.cost_usd < 0:
        return False, "negative_accounting"
    for k, p in decision.probabilities.items():
        if (
            not isinstance(k, str)
            or not isinstance(p, (int, float))
            or not math.isfinite(float(p))
            or not 0 <= float(p) <= 1
        ):
            return False, "invalid_probability"
    if decision.probabilities and not math.isclose(sum(decision.probabilities.values()), 1.0, abs_tol=1e-5):
        return False, "invalid_probability_mass"
    if decision.abstained:
        return True, "provider_abstained"
    if request.kind == DecisionKind.BOOLEAN:
        return (isinstance(decision.value, bool), "invalid_boolean" if not isinstance(decision.value, bool) else "")
    if request.kind in {DecisionKind.CHOICE, DecisionKind.SCORE}:
        ok = any(type(decision.value) is type(option) and decision.value == option for option in request.options)
        return ok, "value_not_in_options" if not ok else ""
    return False, "unsupported_kind"


class BoundedExecutor:
    """No executor queue: timed-out workers retain their slot until they finish.

    Python cannot kill arbitrary thread functions. Adapters must provide their own
    I/O timeout; use process-isolated serving for untrusted/blocking model code.
    """

    def __init__(self, workers: int):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reflex")
        self.slots = threading.BoundedSemaphore(workers)

    async def call(self, fn: Any, *args: Any) -> Any:
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("worker_capacity_exhausted")
        try:
            future = self.pool.submit(fn, *args)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        return await asyncio.shield(wrapped)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


async def _invoke_provider(provider: Any, request: DecisionRequest, executor: BoundedExecutor) -> Decision | None:
    request = deepcopy(request)
    adecide = getattr(provider, "adecide", None)
    if adecide is not None:
        result = adecide(request)
        return cast(Decision | None, await result if inspect.isawaitable(result) else result)
    decide = provider.decide
    if inspect.iscoroutinefunction(decide):
        return cast(Decision | None, await decide(request))
    return cast(Decision | None, await executor.call(decide, request))


@dataclass
class _Accounting:
    attempts: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
    calls: int = 0


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
        policies: Iterable[Policy] = (),
        max_concurrency: int = 64,
        sync_workers: int = 8,
    ):
        if type(max_concurrency) is not int or max_concurrency < 1 or type(sync_workers) is not int or sync_workers < 1:
            raise ValueError("concurrency limits must be positive integers")
        self.policies = tuple(policies)
        self._admission = threading.BoundedSemaphore(max_concurrency)
        self._executor = BoundedExecutor(sync_workers)
        self._inflight = 0
        self._closed = False
        self._lifecycle_lock = threading.Lock()
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
        accounting = _Accounting()
        with self._lifecycle_lock:
            if self._closed or not self._admission.acquire(blocking=False):
                return Decision(
                    None,
                    0,
                    "abstain",
                    abstained=True,
                    metadata={"reason": "engine_closed" if self._closed else "overloaded", "provider_calls": 0},
                )
            self._inflight += 1
        try:
            async with asyncio.timeout(budget.deadline_ms / 1000):
                for policy in self.policies:
                    verdict = await self._executor.call(policy.authorize, deepcopy(request), None)
                    if not isinstance(verdict, PolicyVerdict) or verdict.allow is not True:
                        return self._policy_denied(request, started, accounting)
                result = await self._cascade(
                    request,
                    DecisionBudget(
                        deadline_ms=max(0.001, budget.deadline_ms - (perf_counter() - started) * 1000),
                        max_cost_usd=budget.max_cost_usd,
                        max_provider_calls=budget.max_provider_calls,
                    ),
                    accounting,
                )
                if not result.abstained:
                    for policy in self.policies:
                        verdict = await self._executor.call(policy.authorize, deepcopy(request), result.value)
                        if not isinstance(verdict, PolicyVerdict) or verdict.allow is not True:
                            return self._policy_denied(request, started, accounting)
                result.latency_ms = (perf_counter() - started) * 1000
                reason = result.metadata.get("reason", "forced_best" if result.metadata.get("forced") else "accepted")
                self._emit(request, result, started, len(accounting.attempts), reason)
                return result
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if accounting.attempts and accounting.attempts[-1].get("skipped") == "cancelled":
                accounting.attempts[-1]["skipped"] = "timeout"
            result = Decision(
                None,
                0,
                "abstain",
                abstained=True,
                latency_ms=(perf_counter() - started) * 1000,
                cost_usd=accounting.cost,
                metadata={
                    "reason": "deadline_exhausted",
                    "provider_calls": accounting.calls,
                    "attempts": accounting.attempts,
                    "total_cost_usd": accounting.cost,
                },
            )
            self._emit(request, result, started, len(accounting.attempts), "deadline_exhausted")
            return result
        except Exception:
            # A mandatory policy error fails closed, including an exhausted worker pool.
            return self._policy_denied(request, started, accounting)
        finally:
            with self._lifecycle_lock:
                self._inflight -= 1
                self._admission.release()

    def _policy_denied(self, request: DecisionRequest, started: float, accounting: _Accounting) -> Decision:
        result = Decision(
            None,
            0,
            "policy",
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            cost_usd=accounting.cost,
            metadata={
                "reason": "policy_denied",
                "provider_calls": accounting.calls,
                "attempts": accounting.attempts,
                "total_cost_usd": accounting.cost,
            },
        )
        self._emit(request, result, started, len(accounting.attempts), "policy_denied")
        return result

    async def aclose(self, timeout_s: float = 1.0) -> None:
        self._closed = True
        end = monotonic() + timeout_s
        while self._inflight and monotonic() < end:
            await asyncio.sleep(0.001)
        self._executor.close()

    async def _cascade(self, request: DecisionRequest, budget: DecisionBudget, accounting: _Accounting) -> Decision:
        request.validate()
        budget = budget or self.default_budget
        budget.validate()
        started = perf_counter()
        deadline = monotonic() + budget.deadline_ms / 1000
        attempts = accounting.attempts
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
            if not stage.breaker.available():
                attempts.append({"provider": stage.provider.name, "skipped": "circuit_open"})
                continue
            if provider_calls >= budget.max_provider_calls:
                terminal_reason = "provider_call_budget_exhausted"
                break
            estimated = 0.0
            estimate = getattr(stage.provider, "estimate_cost", None)
            if callable(estimate):
                try:
                    estimated = float(estimate(request))
                except Exception:
                    estimated = float("nan")
                if not math.isfinite(estimated) or estimated < 0:
                    attempts.append({"provider": stage.provider.name, "skipped": "invalid_cost_estimate"})
                    continue
                if total_cost + max(0.0, estimated) > budget.max_cost_usd:
                    attempts.append({"provider": stage.provider.name, "skipped": "cost_budget"})
                    continue

            remaining_ms = max(0.0, (deadline - monotonic()) * 1000)
            if remaining_ms <= 0:
                terminal_reason = "deadline_exhausted"
                break
            timeout_s = min(stage.timeout_ms, remaining_ms) / 1000
            provider_request = (
                self.context_selector.select(request) if stage.sanitize_context and not caps.deterministic else request
            )

            decision: Decision | None = None
            last_error = ""
            for retry_idx in range(stage.retries + 1):
                remaining_ms = max(0.0, (deadline - monotonic()) * 1000)
                if remaining_ms <= 0:
                    terminal_reason = "deadline_exhausted"
                    break
                timeout_s = min(stage.timeout_ms, remaining_ms) / 1000
                if provider_calls >= budget.max_provider_calls:
                    terminal_reason = "provider_call_budget_exhausted"
                    break
                if total_cost + estimated > budget.max_cost_usd:
                    terminal_reason = "cost_budget_exhausted"
                    break
                if not stage.breaker.allow():
                    last_error = "circuit_open"
                    break
                total_cost += estimated
                provider_calls += 1
                accounting.cost = total_cost
                accounting.calls = provider_calls
                try:
                    decision = await asyncio.wait_for(
                        _invoke_provider(stage.provider, provider_request, self._executor), timeout=timeout_s
                    )
                    break
                except TimeoutError:
                    last_error = "timeout"
                    stage.breaker.failure()
                except asyncio.CancelledError:
                    stage.breaker.failure()
                    attempts.append(
                        {
                            "provider": stage.provider.name,
                            "skipped": "timeout" if monotonic() >= deadline else "cancelled",
                        }
                    )
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
                if not last_error:
                    stage.breaker.success()
                attempts.append({"provider": stage.provider.name, "skipped": last_error or "unsupported"})
                continue

            # A malformed response may still represent a billed call.
            if (
                isinstance(decision, Decision)
                and isinstance(decision.cost_usd, (int, float))
                and math.isfinite(decision.cost_usd)
                and decision.cost_usd >= 0
            ):
                total_cost += max(0.0, decision.cost_usd - estimated)
                accounting.cost = total_cost
            valid, reason = _valid_value(request, decision)
            if not valid:
                attempts.append({"provider": stage.provider.name, "skipped": "invalid_typed_output", "reason": reason})
                stage.breaker.failure()
                continue
            try:
                ood = float(decision.metadata.get("ood_score", 0.0))
            except (ValueError, TypeError):
                attempts.append({"provider": stage.provider.name, "skipped": "invalid_ood_score"})
                stage.breaker.failure()
                continue
            if not math.isfinite(ood) or not 0 <= ood <= 1:
                attempts.append({"provider": decision.provider, "skipped": "invalid_ood_score"})
                stage.breaker.failure()
                continue
            stage.breaker.success()
            if total_cost > budget.max_cost_usd:
                attempts.append({"provider": decision.provider, "skipped": "cost_budget_exceeded_after_call"})
                terminal_reason = "cost_budget_exhausted"
                break

            accepted = not decision.abstained and decision.confidence >= stage.threshold and ood <= stage.max_ood
            attempts.append(
                {
                    "provider": decision.provider,
                    "confidence": decision.confidence,
                    "threshold": stage.threshold,
                    "ood_score": ood,
                    "max_ood": stage.max_ood,
                    "value": decision.value,
                    "accepted": accepted,
                    "latency_ms": decision.latency_ms,
                    "cost_usd": decision.cost_usd,
                }
            )
            if (
                not decision.abstained
                and ood <= stage.max_ood
                and (best is None or decision.confidence > best.confidence)
            ):
                best = decision
            if accepted:
                decision.metadata = {
                    **decision.metadata,
                    "attempts": attempts,
                    "total_cost_usd": total_cost,
                    "provider_calls": provider_calls,
                }
                decision.metadata["provider_latency_ms"] = decision.latency_ms
                decision.latency_ms = (perf_counter() - started) * 1000
                decision.cost_usd = total_cost
                return decision

        if best is not None and not self.abstain_on_uncertain and request.risk < self.fail_closed_risk:
            best.metadata = {
                **best.metadata,
                "attempts": attempts,
                "forced": True,
                "total_cost_usd": total_cost,
                "provider_calls": provider_calls,
            }
            best.latency_ms = (perf_counter() - started) * 1000
            best.cost_usd = total_cost
            return best
        result = Decision(
            value=None,
            confidence=best.confidence if best else 0.0,
            provider="abstain",
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            cost_usd=total_cost,
            metadata={
                "attempts": attempts,
                "best_provider": best.provider if best else None,
                "reason": terminal_reason,
                "provider_calls": provider_calls,
            },
        )
        return result

    def decide(self, request: DecisionRequest, budget: DecisionBudget | None = None) -> Decision:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.adecide(request, budget))
        raise RuntimeError("ReflexEngine.decide() cannot run inside an active event loop; use await adecide()")

    async def adecide_many(
        self,
        requests: Iterable[DecisionRequest],
        budget: DecisionBudget | None = None,
        *,
        concurrency: int = 32,
        max_items: int = 10_000,
    ) -> list[Decision]:
        if (
            type(concurrency) is not int
            or not 1 <= concurrency <= 1024
            or type(max_items) is not int
            or not 1 <= max_items <= 100_000
        ):
            raise ValueError("invalid batch limits")
        rows = list(islice(requests, max_items + 1))
        if len(rows) > max_items:
            raise ValueError("batch exceeds max_items")
        results: list[Decision | None] = [None] * len(rows)
        workers = min(concurrency, len(rows))

        async def worker(offset: int) -> None:
            for i in range(offset, len(rows), workers):
                results[i] = await self.adecide(rows[i], budget)

        tasks = [asyncio.create_task(worker(i)) for i in range(workers)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return cast(list[Decision], results)

    def probe_all(self, request: DecisionRequest) -> list[Attempt]:
        return asyncio.run(self.aprobe_all(request))

    async def aprobe_all(self, request: DecisionRequest) -> list[Attempt]:
        request.validate()
        out: list[Attempt] = []
        for stage in self.stages:
            if request.risk > stage.max_risk:
                continue
            try:
                d = await asyncio.wait_for(
                    _invoke_provider(stage.provider, request, self._executor), timeout=stage.timeout_ms / 1000
                )
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

    def health(self) -> dict[str, Any]:
        stages = []
        available = 0
        for stage in self.stages:
            allowed = stage.breaker.available()
            if allowed:
                available += 1
            stages.append(
                {"provider": stage.provider.name, "circuit_available": allowed, "failures": stage.breaker.failures}
            )
        return {
            "ready": not self._closed and bool(self.stages) and available > 0,
            "stages": stages,
            "inflight": self._inflight,
        }

    def _emit(self, request: DecisionRequest, decision: Decision, started: float, attempts: int, reason: str) -> None:
        with suppress(Exception):
            self.telemetry.emit(
                DecisionEvent(
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
                )
            )
