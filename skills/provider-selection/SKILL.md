---
name: provider-selection
description: Select the cheapest safe decision provider for an agent decision using deterministic policy first, then semantic/local inference, then stronger fallback only when uncertainty remains.
---
# Provider selection

1. Classify the decision as deterministic, semantic-bounded, or reasoning-heavy.
2. Use hard rules or policy for deterministic and authorization decisions.
3. Use a local semantic/decision provider only when the state requires fuzzy judgment.
4. Escalate only when confidence, OOD, deadline, risk, or provider-health gates require it.
5. Never let a probabilistic provider override hard policy.
6. Record provider/version, calibrated confidence, OOD, latency, cost, and fallback reason.
