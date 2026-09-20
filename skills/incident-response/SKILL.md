---
name: incident-response
description: Contain and diagnose decision-runtime incidents involving unsafe actions, policy bypass, provider drift, latency spikes, cost spikes, auth failures, or secret leakage.
---
# Incident response

1. Disable unsafe auto-execution or raise the affected threshold immediately.
2. Preserve sanitized decision traces, provider/model versions, policy bundle version, and request IDs.
3. Determine whether the fault came from policy, routing, calibration, provider drift, context selection, auth, or infrastructure.
4. Reproduce against a frozen case and add it to regression/adversarial tests.
5. Ship through offline eval, shadow/canary, then normal rollout; roll back on high-risk regressions.
