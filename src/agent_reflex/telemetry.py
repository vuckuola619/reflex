from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any, Protocol

from agent_reflex.security import sanitize


@dataclass(slots=True)
class DecisionEvent:
    request_id: str
    kind: str
    provider: str
    value: Any
    confidence: float
    abstained: bool
    latency_ms: float
    cost_usd: float
    risk: float
    attempts: int
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    emitted_at: float = field(default_factory=time)


class TelemetrySink(Protocol):
    def emit(self, event: DecisionEvent) -> None: ...


class NoopTelemetrySink:
    def emit(self, event: DecisionEvent) -> None:
        return None


class JsonlTelemetrySink:
    """Thread-safe local audit sink. Sensitive-looking values are redacted."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: DecisionEvent) -> None:
        payload = sanitize(asdict(event))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class OpenTelemetrySink:
    """Optional metrics/tracing bridge; importing the core package never requires OTel."""

    def __init__(self, meter_name: str = "agent_reflex", tracer_name: str = "agent_reflex"):
        try:
            from opentelemetry import metrics, trace
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install agent-reflex[otel]") from exc
        self._tracer = trace.get_tracer(tracer_name)
        meter = metrics.get_meter(meter_name)
        self._decisions = meter.create_counter("agent_reflex.decisions")
        self._latency = meter.create_histogram("agent_reflex.decision.latency_ms", unit="ms")
        self._cost = meter.create_counter("agent_reflex.decision.cost_usd", unit="USD")
        self._abstains = meter.create_counter("agent_reflex.abstains")

    def emit(self, event: DecisionEvent) -> None:  # pragma: no cover - integration adapter
        attrs = {
            "decision.kind": event.kind,
            "decision.provider": event.provider,
            "decision.abstained": event.abstained,
            "decision.risk_bucket": int(event.risk * 10),
        }
        self._decisions.add(1, attrs)
        self._latency.record(event.latency_ms, attrs)
        if event.cost_usd:
            self._cost.add(event.cost_usd, attrs)
        if event.abstained:
            self._abstains.add(1, attrs)
        with self._tracer.start_as_current_span("agent_reflex.decide") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            span.set_attribute("decision.confidence", event.confidence)
            span.set_attribute("decision.attempts", event.attempts)
