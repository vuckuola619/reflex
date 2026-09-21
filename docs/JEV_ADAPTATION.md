# Jev-inspired iteration — 2026-09-21

> The live comparison subsequently completed after approval. See [JEV_COMPARISON.md](JEV_COMPARISON.md) for the current results; approval-pending statements below describe this earlier offline iteration.

This iteration improves local verification and runtime reliability. It does not establish that Reflex is a generally superior semantic model.

## Design lessons adopted

[TypeSafe's design notes](https://typesafe.ai/blog/introducing-system-one-models-and-jev) emphasize typed decisions, probability-aware branching, small questions and workflow evaluation. [Pydantic's Jev integration](https://pydantic.dev/docs/ai/models/typesafe/) demonstrates confidence-based escalation and limiting context sent to the remote service. These are public design ideas; we have not reproduced Jev's architecture or proprietary training.

Reflex now has an offline default (`Reflex.local`) for explicit facts; operator-configured predicates never infer success from prose. Mandatory allowlist/OPA policies run around the cascade. Semantic routing uses pinned MiniLM weights, task scoping, strict label types, similarity and margin rejection, and reports its uncalibrated score provenance.

## Measured local results

Dataset: 12 actual repository commands and five loopback HTTP probes, frozen before this comparison. It includes failed feature checks and fault-injection gates. It is a convenience sample from one repository; observations are correlated and not representative of general production traffic.

SHA256: `226ce00fa0a11705fbcb17874121b0bfa041c907adb482403a81159c10b916c6`.

| Provider | Correct / total | Abstentions | Wall p95 | External calls |
|---|---:|---:|---:|---:|
| Token-overlap prototype baseline | 1 / 17 | 16 | 2.08 ms | 0 |
| Pinned MiniLM semantic router | 1 / 17 | 16 | 205.04 ms | 0 |
| Exact structured evidence checks | 17 / 17 | 0 | 1.12 ms | 0 |

All 17 evidence answers were unchanged under appended instructions to report success. Semantic routing abstained on all 17 perturbed cases. Perturbations are stress replays, not additional independent samples. The exact checks implement the predicates stated in the questions; their advantage is a harness optimization, not evidence of superior language understanding.

The embedding router used 12 synthetic training fixtures and eight separate synthetic validation fixtures. Acceptance parameters were selected on validation and frozen before holdout evaluation. Only four validation cases were accepted without error. Those numbers do not establish calibration. Cold model import/load took 10.66 seconds; the latency table measures subsequent individual calls. Model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The local test ran with remote model access disabled after the initial public-model download.

Outputs: `artifacts/real-comparison-v1/report.json`, per-case JSONL, development fixtures and frozen configuration. Costs exclude local CPU, memory and electricity. The new live Jev comparison has not run: automatic approval review requires specific authorization to send repository outputs. Its exact 42-request payload is `artifacts/jev-review-payload.jsonl` (49,232 bytes, no exact .env secret matches). The benchmark verifies outgoing request bodies against that reviewed payload. Do not compare these measurements to Jev's different eight-case historical smoke.

## Runtime changes and fault drill

- Preserved estimated spend and attempt history across deadline expiration, failed retries and malformed paid responses.
- Emit final decision telemetry after mandatory post-decision policy checks.
- Bounded provider threads and request admission; batch evaluation now creates a bounded worker set and caps input consumption.
- Single circuit-breaker recovery probe; readiness checks do not consume it.
- Strict OPA allow booleans, bounded responses, remote HTTPS and no redirected credentials.
- Correct Jev score probability labels and preserve numeric choice types; out-of-range scores are rejected.
- Fixed the demo's read-only Git rule to require exact commands, avoiding prefix-based approval of compound commands.

`scripts/fire_drill_live.py` exercised actual loopback sockets with controlled HTTP responses: healthy reply, 503 outage, malformed JSON, 300 ms stall, then circuit opening and recovery. All five passed. The stalled provider fell back to local evidence in 45.93 ms. These are real network fault fixtures, not a claim that the real Jev service failed.

Validation: 133 tests passed; 91.06% branch-aware core coverage, now enforced at 90%. Lint and strict source typing passed. Installed dependencies had no known advisories in the current audit, and `pip check` passed. Optional CLI/MCP/Pydantic-SDK/causal-model adapters remain outside core coverage. No production release/tag has been made.

Wheel/sdist builds and metadata validation passed, as did an isolated installed-wheel offline decision. REST readiness, a real MCP client request and a browser-submitted contradictory-log check passed after reloading the services. Source/package scans found no exact .env credential values. Bandit still reports five low-severity findings (bootstrap pseudorandomness, exception handling in probe/SDK paths, and a tuning assertion); no medium/high findings were reported. These checks do not constitute a complete security audit.

## Reproduction and next gate

```powershell
.venv\Scripts\python.exe -m pytest --cov=agent_reflex --cov-branch
.venv\Scripts\python.exe scripts/fire_drill_live.py
.venv\Scripts\python.exe benchmarks/real_workload.py --output artifacts/local-recheck
```

On older CPUs/VMs, install semantic dependencies with `-c constraints/windows-legacy-cpu.txt`; NumPy 2.2.6 worked here. Newer NumPy wheels could not load on this VM. Other machines should use their supported dependency resolution.

After explicit approval of the reviewed payload, run `benchmarks/real_workload.py --live-jev --output artifacts/real-comparison-v2`. It records requested/observed model versions, raw paired answers, failures, latency, estimated cost and stress separately. This small workload still cannot satisfy a general superiority release gate. A larger independently labeled, task-diverse workload and sustained deployment tests remain necessary.
