# Changelog

## 1.0.0rc2

- Added dependency-free direct TypeSafe System One HTTP provider with secret-safe auth handling.
- Pinned Jev benchmark default to `jev-1.13.0`; records requested and observed model ids.
- Fail-closed benchmark validity: provider failures, timeouts, circuit-open baselines, missing model ids and pinned-version mismatches invalidate comparison instead of counting as Jev errors.
- Benchmark latency is now end-to-end wall time and benchmark cost includes total cascade cost.
- Added `scripts/benchmark_jev.sh` one-command paired benchmark + evidence gate.
- Expanded benchmark-integrity regression tests.

## 1.0.0rc1

- Async, bounded decision cascade with deadlines, cost/call budgets, retries, circuit breakers and batch concurrency.
- Typed output firewall with finite-value checks, OOD gating and high-risk fail-closed behavior.
- Secret redaction, bounded context selector, JSONL/OpenTelemetry sinks.
- Deterministic rules, OPA policy adapter, semantic/prototype routing, local Transformers non-generative scoring, Jev benchmark adapter, calibration wrapper.
- REST API factory with authentication, body limits and rate limiting.
- MCP v2/2026-07-28 server surface.
- Calibration/risk-coverage metrics and paired bootstrap comparison support.
- CI, security audit, CodeQL, PyPI Trusted Publishing and Sigstore signing workflows.
- Docker/Compose and Kubernetes deployment examples.
