"""Task-scoped semantic flow with explicit context and validation-fitted gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

from agent_reflex.providers.base import DecisionProvider, ProviderCapabilities, capabilities
from agent_reflex.providers.embedding import EmbeddingExample, EmbeddingRouterProvider, Encoder
from agent_reflex.security import redact_text
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


@dataclass(frozen=True)
class SemanticTask:
    task_id: str
    question: str
    labels: tuple[str, ...]
    fields: tuple[str, ...] = ("summary",)
    max_chars: int = 2048

    def __post_init__(self) -> None:
        if not self.task_id or not self.question.strip() or not 2 <= len(self.labels) <= 64:
            raise ValueError("task needs an id, question and 2..64 labels")
        if any(not isinstance(x, str) or not x for x in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique nonempty strings")
        if not self.fields or len(set(self.fields)) != len(self.fields) or any(not f for f in self.fields):
            raise ValueError("fields must be unique and nonempty")
        if type(self.max_chars) is not int or not 32 <= self.max_chars <= 8192:
            raise ValueError("max_chars must be in 32..8192")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _distribution(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not probabilities or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature or empty distribution")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or not math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-5
    ):
        raise ValueError("invalid probability distribution")
    scores = {key: math.log(max(value, 1e-12)) / temperature for key, value in probabilities.items()}
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


@dataclass(frozen=True)
class SemanticProfile:
    """A development artifact, not a guarantee of population error rates."""

    fingerprint: str
    temperature: float = 1.0
    min_confidence: float = 0.8
    min_similarity: float = 0.3
    min_margin: float = 0.05
    enabled: bool = False
    calibration_sha256: str = ""
    validation_sha256: str = ""
    validation_count: int = 0
    validation_accepted: int = 0
    validation_errors: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid calibration temperature")
        if not 0 <= self.min_confidence <= 1 or not -1 <= self.min_similarity <= 1 or not 0 <= self.min_margin <= 2:
            raise ValueError("invalid semantic gates")


class SemanticFlow:
    """One registered decision task. Missing/foreign context fails closed.

    Use separate instances (sharing the encoder) for distinct tasks. Local scores
    remain disabled until a matching validation profile is fitted or supplied.
    A task id is a routing hint, never authorization. Keep hard policies outside.
    """

    name = "semantic-flow"
    capabilities = ProviderCapabilities(kinds=frozenset({DecisionKind.CHOICE}), local=True, probabilities=True)

    def __init__(
        self,
        task: SemanticTask,
        examples: Sequence[EmbeddingExample],
        *,
        encoder: Encoder | None = None,
        profile: SemanticProfile | None = None,
        local_files_only: bool = True,
    ):
        self.task = task
        if any(type(ex.label) is not str or ex.label not in task.labels for ex in examples):
            raise ValueError("training label outside task contract")
        if any(sum(ex.label == label for ex in examples) < 2 for label in task.labels):
            raise ValueError("each class needs at least two training examples")
        texts = [ex.text.strip() for ex in examples]
        if any(not t or len(t) > task.max_chars for t in texts) or len(set(texts)) != len(texts):
            raise ValueError("training texts must be unique, nonempty and bounded")
        self.training_texts = frozenset(texts)
        self.router = EmbeddingRouterProvider(
            [EmbeddingExample(redact_text(ex.text.strip()), ex.label) for ex in examples],
            temperature=1,
            min_similarity=-1,
            min_margin=0,
            encoder=encoder,
            local_files_only=local_files_only,
        )
        self.fingerprint = _digest(
            {
                "task": asdict(task),
                "examples": [asdict(e) for e in examples],
                "model": self.router.model_name,
                "revision": self.router.revision,
            }
        )
        self.profile = profile or SemanticProfile(self.fingerprint)
        if self.profile.fingerprint != self.fingerprint:
            raise ValueError("profile does not match task, training data and model")

    def prepare(self, request: DecisionRequest) -> tuple[DecisionRequest | None, str]:
        if request.metadata.get("task") != self.task.task_id:
            return None, "unknown_task"
        if request.kind != DecisionKind.CHOICE or request.question != self.task.question:
            return None, "question_contract_mismatch"
        if (
            any(type(x) is not str for x in request.options)
            or len(request.options) != len(self.task.labels)
            or set(request.options) != set(self.task.labels)
        ):
            return None, "option_contract_mismatch"
        if not isinstance(request.state, dict):
            return None, "structured_context_required"
        selected = []
        for field in self.task.fields:
            text = request.state.get(field)
            if not isinstance(text, str) or not text.strip():
                return None, "required_context_missing"
            if len(text) > self.task.max_chars or "[TRUNCATED" in text:
                return None, "context_too_long"
            selected.append(redact_text(text.strip()))
        text = "\n".join(selected)
        if len(text) > self.task.max_chars:
            return None, "context_too_long"
        # Do not leak metadata, unselected fields or caller-owned context to fallback.
        return replace(
            request,
            state=text,
            options=self.task.labels,
            metadata={"task": self.task.task_id},
            trusted_context={},
            untrusted_context={},
        ), ""

    def _abstain(self, reason: str, started: float) -> Decision:
        return Decision(
            None,
            0,
            self.name,
            abstained=True,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={"reason": reason, "task": self.task.task_id, "profile": self.fingerprint},
        )

    def raw(self, request: DecisionRequest) -> Decision | None:
        selected, _ = self.prepare(request)
        return self.router.decide(selected) if selected is not None else None

    def decide(self, request: DecisionRequest) -> Decision:
        started = perf_counter()
        selected, reason = self.prepare(request)
        if selected is None:
            return self._abstain(reason, started)
        if not self.profile.enabled:
            return self._abstain("profile_not_validated", started)
        raw = self.router.decide(selected)
        if raw is None:
            return self._abstain("no_semantic_evidence", started)
        probabilities = _distribution(raw.probabilities, self.profile.temperature)
        confidence = probabilities[str(raw.value)]
        accepted = (
            confidence >= self.profile.min_confidence
            and raw.metadata["similarity"] >= self.profile.min_similarity
            and raw.metadata["margin"] >= self.profile.min_margin
        )
        return Decision(
            raw.value if accepted else None,
            confidence,
            self.name,
            probabilities=probabilities,
            abstained=not accepted,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={
                **raw.metadata,
                "reason": "semantic_accepted" if accepted else "semantic_uncertain",
                "task": self.task.task_id,
                "profile": self.fingerprint,
                "confidence_kind": "validation_fitted_temperature",
                "calibration_sha256": self.profile.calibration_sha256,
            },
        )

    def fit(
        self,
        calibration: Sequence[tuple[str, str]],
        validation: Sequence[tuple[str, str | None]],
        *,
        min_accepted: int = 4,
    ) -> SemanticProfile:
        """Fit temperature on calibration, gates on validation; never pass holdout rows.

        None labels in validation represent out-of-domain inputs that must abstain.
        The policy allows zero observed validation errors; small sets do not prove
        a population risk bound. An infeasible validation set disables the stage.
        """
        if not calibration or not validation or type(min_accepted) is not int or min_accepted < 1:
            raise ValueError("nonempty development splits and positive min_accepted required")
        cal_texts = [t.strip() for t, _ in calibration]
        val_texts = [t.strip() for t, _ in validation]
        all_texts = cal_texts + val_texts
        if (
            any(not t or len(t) > self.task.max_chars for t in all_texts)
            or len(set(all_texts)) != len(all_texts)
            or self.training_texts.intersection(all_texts)
        ):
            raise ValueError("development splits must be nonempty, bounded and disjoint")
        if any(y not in self.task.labels for _, y in calibration) or any(
            y is not None and y not in self.task.labels for _, y in validation
        ):
            raise ValueError("development label outside task")

        def probe(text: str) -> Decision:
            result = self.router.decide(
                DecisionRequest(redact_text(text.strip()), self.task.question, DecisionKind.CHOICE, self.task.labels)
            )
            if result is None:
                raise ValueError("cannot calibrate empty semantic prediction")
            return result

        cal = [probe(t) for t, _ in calibration]
        temperatures = (0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0)
        temperature = min(
            temperatures,
            key=lambda t: sum(
                -math.log(max(1e-12, _distribution(d.probabilities, t)[gold]))
                for d, (_, gold) in zip(cal, calibration, strict=True)
            ),
        )
        vals = [probe(t) for t, _ in validation]
        candidates = []
        for conf in (0.5, 0.65, 0.8, 0.9, 0.95):
            for similarity in (0.25, 0.4, 0.55, 0.7):
                for margin in (0.03, 0.08, 0.15, 0.25):
                    chosen = [
                        (d, gold)
                        for d, (_, gold) in zip(vals, validation, strict=True)
                        if _distribution(d.probabilities, temperature)[str(d.value)] >= conf
                        and d.metadata["similarity"] >= similarity
                        and d.metadata["margin"] >= margin
                    ]
                    if len(chosen) >= min_accepted and all(gold is not None and d.value == gold for d, gold in chosen):
                        candidates.append((len(chosen), conf, similarity, margin))
        accepted, conf, similarity, margin = max(candidates, default=(0, 1.0, 1.0, 2.0))
        self.profile = SemanticProfile(
            self.fingerprint,
            temperature,
            conf,
            similarity,
            margin,
            bool(candidates),
            _digest(calibration),
            _digest(validation),
            len(validation),
            accepted,
            0,
        )
        return self.profile


class SemanticFallback:
    """Apply the same task contract/context selection before an optional fallback.

    The outer ReflexEngine owns deadlines, spend, policy and output validation.
    This sync adapter accepts sync providers such as DirectJevProvider.
    """

    def __init__(self, flow: SemanticFlow, provider: DecisionProvider):
        self.flow = flow
        self.provider = provider
        self.name = f"{provider.name}:scoped"
        self.capabilities = capabilities(provider)

    def estimate_cost(self, request: DecisionRequest) -> float:
        selected, _ = self.flow.prepare(request)
        estimate = getattr(self.provider, "estimate_cost", None)
        if selected is None:
            return 0.0
        if not callable(estimate):
            return 0.0 if self.capabilities.local else float("nan")
        return float(estimate(selected))

    def decide(self, request: DecisionRequest) -> Decision | None:
        selected, _ = self.flow.prepare(request)
        if selected is None:
            return None
        result = self.provider.decide(selected)
        if result is not None:
            result = replace(result, metadata={**result.metadata, "fallback_task": self.flow.task.task_id})
        return result
