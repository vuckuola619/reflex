from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


@dataclass(slots=True)
class EmbeddingExample:
    text: str
    label: Any


class EmbeddingRouterProvider:
    name = "semantic"
    capabilities = ProviderCapabilities(local=True, probabilities=True)

    def __init__(self, examples: Iterable[EmbeddingExample], model_name: str = "all-MiniLM-L6-v2", *, include_question: bool = False, temperature: float = 0.08):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install agent-reflex[semantic]") from exc
        self.np=np
        self.model=SentenceTransformer(model_name)
        self.examples=list(examples)
        self.include_question=include_question
        self.temperature=max(float(temperature),1e-6)
        if not self.examples:
            raise ValueError("at least one embedding example is required")
        self.embeddings=self.model.encode([x.text for x in self.examples], normalize_embeddings=True, show_progress_bar=False)

    def decide(self, request: DecisionRequest) -> Decision | None:
        started=perf_counter()
        text=str(request.state)
        if self.include_question:
            text=f"{request.question}\n{text}"
        q=self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims=self.embeddings @ q
        per_label: dict[Any,float]={}
        for ex,sim in zip(self.examples,sims):
            if request.kind in {DecisionKind.CHOICE,DecisionKind.SCORE} and ex.label not in request.options:
                continue
            if request.kind==DecisionKind.BOOLEAN and not isinstance(ex.label,bool):
                continue
            per_label[ex.label]=max(float(sim),per_label.get(ex.label,-1.0))
        if not per_label:
            return None
        vals=self.np.array(list(per_label.values()),dtype=float)
        logits=vals/self.temperature
        logits-=logits.max()
        probs_arr=self.np.exp(logits); probs_arr/=probs_arr.sum()
        labels=list(per_label)
        probs={str(k):float(v) for k,v in zip(labels,probs_arr)}
        value=max(labels,key=lambda x: probs[str(x)])
        sim=max(0.0,min(1.0,(per_label[value]+1.0)/2.0))
        return Decision(value=value, confidence=probs[str(value)], provider=self.name, probabilities=probs,
                        latency_ms=(perf_counter()-started)*1000,
                        metadata={"similarity":per_label[value],"ood_score":1.0-sim})
