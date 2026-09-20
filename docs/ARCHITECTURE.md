# Architecture

Agent Reflex is a bounded decision runtime for agent harnesses. It is intentionally not a text-generation framework.

```text
request
  -> validation + secret-safe context bounding
  -> hard policy / deterministic rules
  -> semantic route
  -> local decision model
  -> optional remote fallback (including Jev)
  -> typed-output firewall
  -> confidence + OOD + risk gate
  -> action / abstain
  -> telemetry + feedback/eval
```

## Core invariants

- Deterministic policy outranks model judgment.
- High-risk work fails closed by abstaining unless a deterministic or explicitly trusted provider resolves it.
- Every provider is bounded by deadline, cost, call-count, typed output, OOD and circuit-breaker gates.
- Provider output is data, not authority. Invalid values, NaN/Inf, out-of-range probabilities and out-of-schema choices are rejected.
- Telemetry is best-effort and cannot break the decision path.
- Untrusted context remains untrusted even after sanitization.

## Why heterogeneous routing

Some decisions are cheaper and more reliable as ordinary code: exit status, file existence, date arithmetic, explicit deny rules. Others need semantic judgment. A single model on every request wastes latency/cost and inherits that model's weaknesses. Agent Reflex routes each decision to the cheapest provider that can safely resolve it and abstains when no provider clears its evidence threshold.
