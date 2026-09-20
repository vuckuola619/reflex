from __future__ import annotations

from time import perf_counter
from typing import Literal

from agent_reflex.providers.base import ProviderCapabilities
from agent_reflex.types import Decision, DecisionKind, DecisionRequest


class JevProvider:
    """Optional Jev adapter for apples-to-apples benchmarking/fallback."""
    name="jev"
    capabilities=ProviderCapabilities(local=False, probabilities=True)

    def __init__(self, model: str="typesafe:jev-latest"):
        self.model=model

    def _agent(self, request: DecisionRequest):
        try:
            from pydantic import Field, create_model
            from pydantic_ai import Agent
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install agent-reflex[jev]") from exc
        if request.kind==DecisionKind.BOOLEAN:
            Output=create_model("BooleanDecision",answer=(bool,Field(description=request.question)))
        else:
            option_type=Literal.__getitem__(tuple(request.options))
            Output=create_model("ChoiceDecision",answer=(option_type,Field(description=request.question)))
        return Agent(self.model,output_type=Output)

    @staticmethod
    def _convert(result, started: float) -> Decision:
        value=getattr(result.output,"answer")
        details=result.response.provider_details or {}
        cmap=details.get("confidence",{})
        confidence=float(cmap.get("answer",cmap.get("response",0.5)))
        probabilities=details.get("probabilities",{}).get("answer",{}) or {}
        cost=0.0
        try:
            if result.response.usage and result.response.usage.cost is not None:
                cost=float(result.response.usage.cost)
        except Exception:
            pass
        return Decision(value=value,confidence=confidence,provider="jev",
                        probabilities={str(k):float(v) for k,v in probabilities.items()},
                        latency_ms=(perf_counter()-started)*1000,cost_usd=cost,
                        metadata={"model":result.response.model_name})

    def decide(self, request: DecisionRequest) -> Decision | None:  # pragma: no cover
        started=perf_counter(); result=self._agent(request).run_sync(str(request.state)); return self._convert(result,started)

    async def adecide(self, request: DecisionRequest) -> Decision | None:  # pragma: no cover
        started=perf_counter(); result=await self._agent(request).run(str(request.state)); return self._convert(result,started)
