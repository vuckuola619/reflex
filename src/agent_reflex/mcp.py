from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server import MCPServer

from agent_reflex.engine import DecisionBudget, ReflexEngine
from agent_reflex.types import DecisionKind, DecisionRequest


def create_mcp_server(engine: ReflexEngine) -> MCPServer:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError("Install agent-reflex[mcp]") from exc
    mcp = MCPServer("Agent Reflex")

    @mcp.tool()
    async def decide_choice(
        state: str,
        question: str,
        options: list[str],
        risk: float = 0.5,
        deadline_ms: float = 1000.0,
        max_cost_usd: float = 0.01,
    ) -> dict[str, Any]:
        """Make a bounded decision. Uncertain/high-risk requests abstain instead of forcing an answer."""
        req = DecisionRequest(state, question, DecisionKind.CHOICE, options, risk)
        d = await engine.adecide(
            req, DecisionBudget(deadline_ms=deadline_ms, max_cost_usd=max_cost_usd, max_provider_calls=4)
        )
        return {
            "value": d.value,
            "confidence": d.confidence,
            "provider": d.provider,
            "abstained": d.abstained,
            "probabilities": d.probabilities,
            "request_id": req.request_id,
        }

    @mcp.tool()
    async def decide_boolean(
        state: str, question: str, risk: float = 0.5, deadline_ms: float = 1000.0
    ) -> dict[str, Any]:
        req = DecisionRequest(state, question, DecisionKind.BOOLEAN, risk=risk)
        d = await engine.adecide(req, DecisionBudget(deadline_ms=deadline_ms))
        return {
            "value": d.value,
            "confidence": d.confidence,
            "provider": d.provider,
            "abstained": d.abstained,
            "request_id": req.request_id,
        }

    return mcp
