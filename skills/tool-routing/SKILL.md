---
name: tool-routing
description: Route an agent request to one bounded tool or abstain, using deterministic signals first and semantic decisions only when needed.
---
# Tool routing

1. Extract only the facts required to choose a tool.
2. If a deterministic rule selects or blocks a tool, use that result.
3. Otherwise call the decision runtime with a `choice` question whose options are the available tool names plus `none`.
4. Require higher confidence for tools with external side effects.
5. If confidence is below the configured threshold, abstain or escalate to the agent's stronger reasoning model.
6. Record the provider, confidence, selected tool, and whether fallback was used.
