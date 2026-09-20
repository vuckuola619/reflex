from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest

_TOKEN = re.compile(r"[a-zA-Z0-9_./:-]+")


def _tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else repr(value)
    return {m.group(0).lower() for m in _TOKEN.finditer(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(slots=True)
class Prototype:
    text: str
    label: Any


class PrototypeProvider:
    """Dependency-free baseline; intended for tests/small installs, not high-stakes semantics."""
    name = "prototype"
    capabilities = ProviderCapabilities(local=True, probabilities=True)

    def __init__(self, examples: Iterable[Prototype] = (), temperature: float = 0.15, *, include_question: bool = False):
        self.examples = list(examples)
        self.temperature = max(temperature, 1e-6)
        self.include_question = include_question
        self._encoded = [(_tokens(x.text), x.label) for x in self.examples]

    def decide(self, request: DecisionRequest) -> Decision | None:
        if not self._encoded:
            return None
        started = perf_counter()
        query = _tokens(request.state)
        if self.include_question:
            query |= _tokens(request.question)
        scores: dict[Any, float] = defaultdict(float)
        for encoded, label in self._encoded:
            scores[label] = max(scores[label], _jaccard(query, encoded))
        if request.kind in {DecisionKind.CHOICE, DecisionKind.SCORE}:
            scores = {k:v for k,v in scores.items() if k in request.options}
        elif request.kind == DecisionKind.BOOLEAN:
            scores = {k:v for k,v in scores.items() if isinstance(k, bool)}
        if not scores:
            return None
        labels=list(scores)
        logits=[scores[x]/self.temperature for x in labels]
        mx=max(logits)
        exps=[math.exp(x-mx) for x in logits]
        total=sum(exps) or 1.0
        probs={str(label): e/total for label,e in zip(labels,exps)}
        value=max(labels,key=lambda x: probs[str(x)])
        sim=max(0.0,min(1.0,float(scores[value])))
        return Decision(value=value, confidence=probs[str(value)], provider=self.name, probabilities=probs,
                        latency_ms=(perf_counter()-started)*1000,
                        metadata={"max_similarity": sim, "ood_score": 1.0-sim})

    async def adecide(self, request: DecisionRequest) -> Decision | None:
        return self.decide(request)
