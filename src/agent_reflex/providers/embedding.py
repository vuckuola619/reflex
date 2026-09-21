from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


@dataclass(slots=True)
class EmbeddingExample:
    text: str
    label: Any


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class EmbeddingRouterProvider:
    """Local routing with explicit similarity/margin rejection and pinned weights.

    Similarity-softmax scores are not calibrated probabilities of correctness.
    Freeze examples and thresholds on development data before evaluating a holdout.
    task_id isolates decision domains; it is a routing hint, not authorization.
    """

    name = "semantic"
    capabilities = ProviderCapabilities(local=True, probabilities=True)

    def __init__(
        self,
        examples: Iterable[EmbeddingExample],
        model_name: str = MINILM_MODEL,
        *,
        include_question: bool = False,
        temperature: float = 0.08,
        revision: str | None = None,
        min_similarity: float = 0.3,
        min_margin: float = 0.05,
        task_id: str | None = None,
        cache_folder: str | None = None,
        local_files_only: bool = False,
        encoder: Encoder | None = None,
    ):
        self.examples = list(examples)
        self.include_question = include_question
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not -1 <= min_similarity <= 1 or not 0 <= min_margin <= 2:
            raise ValueError("invalid semantic acceptance limits")
        self.temperature = temperature
        if not self.examples:
            raise ValueError("at least one embedding example is required")
        self.revision = revision or (MINILM_REVISION if model_name == MINILM_MODEL else None)
        if encoder is None:
            if not self.revision:
                raise ValueError("custom models require a pinned revision")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install agent-reflex[semantic]") from exc
            encoder = SentenceTransformer(
                model_name,
                revision=self.revision,
                trust_remote_code=False,
                device="cpu",
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
        self.model = encoder
        self.model_name = model_name
        self.task_id = task_id
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self._lock = threading.Lock()
        self.embeddings = self._encode([x.text for x in self.examples])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if len(vectors) != len(texts):
            raise ValueError("embedding batch size mismatch")
        result = []
        for vector in vectors:
            values = [float(x) for x in vector]
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError("invalid embedding")
            norm = math.sqrt(sum(x * x for x in values))
            if norm == 0:
                raise ValueError("zero embedding")
            result.append([x / norm for x in values])
        return result

    def decide(self, request: DecisionRequest) -> Decision | None:
        if self.task_id is not None and request.metadata.get("task") != self.task_id:
            return None
        started = perf_counter()
        text = request.state if isinstance(request.state, str) else json.dumps(request.state, sort_keys=True)
        if self.include_question:
            text = f"{request.question}\n{text}"
        q = self._encode([text])[0]
        per_label: dict[tuple[type, str], tuple[Any, float]] = {}
        for ex, vector in zip(self.examples, self.embeddings, strict=True):
            if request.kind in {DecisionKind.CHOICE, DecisionKind.SCORE} and not any(
                type(ex.label) is type(o) and ex.label == o for o in request.options
            ):
                continue
            if request.kind == DecisionKind.BOOLEAN and type(ex.label) is not bool:
                continue
            if len(vector) != len(q):
                raise ValueError("embedding dimension mismatch")
            sim = sum(x * y for x, y in zip(vector, q, strict=True))
            key = (type(ex.label), str(ex.label))
            per_label[key] = (ex.label, max(sim, per_label.get(key, (None, -1.0))[1]))
        ranked = sorted(per_label.values(), key=lambda pair: pair[1], reverse=True)
        if len(ranked) < 2:
            return None
        value, sim = ranked[0]
        margin = sim - ranked[1][1]
        if sim < self.min_similarity or margin < self.min_margin:
            return None
        weights = [math.exp((s - sim) / self.temperature) for _, s in ranked]
        total = sum(weights)
        probs = {str(label): w / total for (label, _), w in zip(ranked, weights, strict=True)}
        if len(probs) != len(ranked):
            return None
        return Decision(
            value=value,
            confidence=probs[str(value)],
            provider=self.name,
            probabilities=probs,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={
                "similarity": sim,
                "margin": margin,
                "ood_score": 1.0 - max(0.0, min(1.0, sim)),
                "model": self.model_name,
                "revision": self.revision,
                "confidence_kind": "uncalibrated_similarity_softmax",
            },
        )
