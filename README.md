# Agent Reflex

**Open, model-agnostic decision runtime for agent harnesses. Deterministic first, semantic second, model last.**

Agent Reflex turns bounded agent judgments into typed decisions with confidence, provenance, budgets, abstention, policy enforcement and measurable failure behavior. It is designed to complement coding/agent models, not replace them.

```text
Agent state / trace
        |
        v
validation + bounded/redacted context
        |
        +--> hard rules / OPA  -------------------+
        |                                         |
        +--> semantic router                      |
        |        | uncertain/OOD                  |
        +--> local decision scorer                |
        |        | uncertain                      |
        +--> optional remote fallback / Jev       |
                                                 v
                                 typed-output firewall
                                           |
                                  confidence/risk gate
                                           |
                                   action or abstain
                                           |
                                telemetry + eval loop
```

## Why

A coding agent does not need a frontier LLM call for every `retry?`, `done?`, `which tests?`, `is this allowed?`, or `which tool?` decision. Many of those are deterministic; many others can be resolved by small/local bounded models. Agent Reflex provides one interface across those layers and escalates only when necessary.

This is intentionally broader than a Jev clone. Jev can be a benchmark or fallback provider; the runtime remains independent of it.

## Current release status

`1.0.0rc2` contains the production-control surface. It is deliberately an RC rather than falsely claiming universal production proof.

Validated in this workspace:

- **38/38 tests pass**
- **>=80% branch-aware core coverage gate** with an 80% CI floor
- typed-output/NaN/probability firewall tests
- timeout, retry, call-budget, cost-budget and circuit-breaker tests
- high-risk fail-closed and OOD tests
- secret-redaction and telemetry-failure isolation tests
- authenticated REST integration tests
- async bounded batch tests
- synthetic train/validation/test benchmark loop
- local 10k-request throughput smoke test

The bundled benchmark is deliberately tiny/synthetic. It is not evidence that Agent Reflex is generally better than Jev. A live paired Jev gate is included and requires a real API key + frozen real-world holdout.

## Install

Core:

```bash
pip install -e .
```

Useful extras:

```bash
pip install -e '.[server]'
pip install -e '.[mcp]'
pip install -e '.[semantic]'
pip install -e '.[local-model]'
pip install -e '.[jev]'  # only needed for the optional Pydantic AI adapter; direct Jev HTTP needs no extra
pip install -e '.[otel]'
```

## Basic usage

```python
from agent_reflex import DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.providers import Rule, RuleProvider

rules = RuleProvider([
    Rule("block-root-delete", lambda r: "rm -rf /" in str(r.state), "deny"),
    Rule("allow-status", lambda r: str(r.state).startswith("git status"), "allow"),
])

engine = ReflexEngine([
    ProviderStage(rules, threshold=1.0, timeout_ms=25, sanitize_context=False),
])

result = engine.decide(
    DecisionRequest(
        state="rm -rf /",
        question="How should this command be handled?",
        kind=DecisionKind.CHOICE,
        options=["allow", "review", "deny"],
        risk=1.0,
    )
)

assert result.value == "deny"
```

Async production path:

```python
result = await engine.adecide(request)
results = await engine.adecide_many(requests, concurrency=32)
```

## Runtime controls

Each cascade has a global `DecisionBudget`:

```python
from agent_reflex import DecisionBudget

budget = DecisionBudget(
    deadline_ms=300,
    max_cost_usd=0.001,
    max_provider_calls=3,
)

result = await engine.adecide(request, budget)
```

Each provider stage can independently define:

```text
confidence threshold
max request risk
max OOD score
provider timeout
retry count
circuit breaker
context sanitization
```

The engine rejects:

```text
NaN / Infinity confidence
confidence outside [0,1]
negative latency/cost
invalid probabilities
values outside typed choices
untrusted high-risk probabilistic decisions
providers above OOD threshold
```

## Provider cascade

Included providers:

| Provider | Purpose |
|---|---|
| `RuleProvider` | deterministic local policy/logic |
| `OpaPolicyProvider` | OPA/Rego hard policy |
| `PrototypeProvider` | zero-dependency routing baseline |
| `EmbeddingRouterProvider` | local embedding routing |
| `TransformersScoringProvider` | non-generative local option scoring |
| `CalibratedProvider` | confidence calibration wrapper |
| `DirectJevProvider` | dependency-free direct TypeSafe System One HTTP benchmark/fallback adapter |
| `JevProvider` | optional Pydantic AI TypeSafe/Jev adapter |

`TransformersScoringProvider` evaluates bounded options by conditional likelihood rather than asking the model to generate prose/JSON.

## Security model

Hard policy always outranks probabilistic judgment. At the default high-risk boundary, ordinary probabilistic providers are skipped and the runtime abstains unless a deterministic/explicitly trusted provider handles the request.

Context is size-bounded and common credential patterns are redacted before probabilistic providers/audit sinks. This is a leakage-reduction measure, **not** a guarantee that arbitrary untrusted input is safe. Keep authorization outside the model and run high-impact policy through deterministic code or OPA.

See [SECURITY.md](SECURITY.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## REST API

REST serving is authenticated by default and refuses to start without API keys unless unauthenticated mode is explicitly enabled.

```bash
export AGENT_REFLEX_API_KEYS='replace-with-a-real-random-key'
agent-reflex serve \
  --factory examples.server_app:build_engine \
  --host 0.0.0.0 \
  --port 8080
```

```http
POST /v1/decide
X-API-Key: ...
X-Tenant-ID: optional-tenant
```

The API includes body-size limits, sliding-window rate limiting, health/readiness probes and per-request decision budgets.

## MCP

The optional MCP surface targets the stable Python SDK v2 and the **2026-07-28** protocol. Modern MCP requests are stateless by protocol design; `stateless_http=True` also makes the legacy path sessionless for horizontal scaling.

```bash
pip install -e '.[mcp]'
agent-reflex mcp --factory examples.server_app:build_engine
```

## OPA sidecar

```bash
export AGENT_REFLEX_API_KEYS='replace-with-a-real-random-key'
docker compose up --build
```

The Compose/Kubernetes examples pin OPA `1.19.0-static` and keep the probabilistic runtime unable to override policy.

## Observability

Use `JsonlTelemetrySink` locally or `OpenTelemetrySink` in an instrumented application. Decision events include request id, provider, confidence, abstention, latency, cost, risk, attempts and sanitized metadata.

Recommended production dashboards/alerts:

```text
p50/p95/p99 latency
provider errors/timeouts
circuit-open count
abstention / fallback rate
OOD rate
policy denies
high-risk decisions
cost per 1k decisions
calibration drift
```

## Calibration and OOD

Confidence must be calibrated against your own labels. The repo includes temperature scaling, ECE, Brier, NLL and risk-coverage curves. Calibration and threshold tuning belong on a validation split; the final holdout stays untouched.

Semantic providers can attach `ood_score`; a stage rejects answers above `max_ood` even when model confidence is high.

## Benchmark / iterate loop

```bash
PYTHONPATH=src python benchmarks/iterate.py --json-out benchmark-local.json
```

The loop performs:

```text
train examples
   -> provider construction
validation
   -> threshold optimization
freeze
   -> untouched test
   -> adversarial transforms
   -> calibration / coverage / cost / latency
```

Run a paired Jev comparison:

```bash
export TYPESAFE_API_KEY='...'
PYTHONPATH=src python benchmarks/iterate.py \
  --dataset your-frozen-real-dataset.jsonl \
  --compare-jev \
  --jev-adapter http \
  --jev-model jev-1.13.0 \
  --json-out benchmark-jev.json

PYTHONPATH=src python benchmarks/gate.py benchmark-jev.json

# or use the fail-fast wrapper
./scripts/benchmark_jev.sh your-frozen-real-dataset.jsonl benchmark-jev.json
```

The report records dataset SHA-256, Python/platform information, the requested Jev model and the model id actually returned by TypeSafe. The gate refuses to compare when any Jev row failed, timed out, abstained because of provider failure, or returned the wrong pinned model. Benchmark latency is end-to-end wall time and cost is total cascade cost, not merely the final provider hop. The gate then requires accuracy/selective-accuracy non-inferiority plus configured calibration, speed, cost and robustness constraints. A passing gate means only that the criteria passed **on that workload**.

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Local performance smoke test

```bash
PYTHONPATH=src python benchmarks/load_local.py --n 10000 --concurrency 64
```

On the workspace used to prepare this RC, a lightweight rules + prototype cascade processed ~18.3k decisions/s over 10k requests at concurrency 64. This is a local microbenchmark; do not extrapolate it to remote models, different hardware, or application-level throughput.

## Agent Skills

Portable skills are under `skills/`:

```text
tool-routing
risk-gate
completion-check
provider-selection
calibration-gate
benchmark-claim
incident-response
```

The skills express reusable procedure/policy; Agent Reflex is the runtime executing bounded decisions underneath them.

## Reproducibility and release

In an internet-connected trusted environment:

```bash
./scripts/lock.sh
```

This generates `uv.lock`, standardized `pylock.toml`, and an optional CycloneDX SBOM. The workspace used for this build had no outbound PyPI DNS, so no fake lockfile is included.

Verification pipeline:

```bash
./scripts/verify.sh
```

GitHub workflows include Python 3.11/3.12/3.13 CI, benchmark smoke, pip-audit, CodeQL, paired Jev benchmark, PyPI OIDC Trusted Publishing, and Sigstore keyless artifact signing.

## What remains before calling a specific deployment "10/10"

The framework controls are implemented, but production readiness is workload-specific. Before promoting `1.0.0rc2` to a deployment-specific stable release, complete these external evidence gates:

1. Freeze and label real agent decision traces representative of your production traffic.
2. Run the paired Jev benchmark against a pinned version (currently documented example: `jev-1.13.0`).
3. Run load/soak/chaos tests on the actual deployment hardware and provider topology.
4. Generate/commit the lockfile and SBOM in connected CI.
5. Run an independent security review for the concrete policies/actions your agent can execute.
6. Canary the new decision policy and automatically roll back on high-risk regressions.

That list is intentionally not papered over: a generic framework cannot honestly certify a user's unknown production workload without the workload.

## References used for the production design

- MCP 2026-07-28 specification and Python SDK v2 deployment guidance
- OpenTelemetry Python instrumentation guidance
- Open Policy Agent REST API, bundles and decision logs
- vLLM structured outputs
- Python Packaging `pylock.toml` / PEP 751
- PyPI Trusted Publishing
- Sigstore Python/keyless signing
- Pydantic AI TypeSafe/Jev model guidance

## License

MIT
