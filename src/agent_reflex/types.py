from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Hashable, Sequence
from uuid import uuid4


class DecisionKind(str, Enum):
    BOOLEAN = "boolean"
    CHOICE = "choice"
    SCORE = "score"


@dataclass(slots=True)
class DecisionRequest:
    state: Any
    question: str
    kind: DecisionKind = DecisionKind.BOOLEAN
    options: Sequence[Hashable] = field(default_factory=tuple)
    risk: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    trusted_context: dict[str, Any] = field(default_factory=dict)
    untrusted_context: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def validate(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be non-empty")
        if len(self.question) > 16_384:
            raise ValueError("question exceeds 16384 characters")
        if not isfinite(float(self.risk)) or not 0 <= float(self.risk) <= 1:
            raise ValueError("risk must be finite and in [0, 1]")
        if self.kind in {DecisionKind.CHOICE, DecisionKind.SCORE}:
            if not self.options:
                raise ValueError(f"{self.kind.value} decisions require options")
            if len(self.options) > 512:
                raise ValueError("options exceed maximum of 512")
            normalized = [repr(x) for x in self.options]
            if len(set(normalized)) != len(normalized):
                raise ValueError("options must be unique")


@dataclass(slots=True)
class Decision:
    value: Any
    confidence: float
    provider: str
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence)
        self.latency_ms = float(self.latency_ms)
        self.cost_usd = float(self.cost_usd)


@dataclass(slots=True)
class Attempt:
    provider: str
    decision: Decision | None
    threshold: float
    accepted: bool
    reason: str = ""
    attempt_number: int = 1
