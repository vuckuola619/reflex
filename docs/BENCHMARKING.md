# Benchmarking and claims

`benchmarks/iterate.py` follows train -> validation threshold tuning -> frozen test. `benchmarks/gate.py` refuses a Jev comparison report unless both systems were run on the same holdout.

Required production benchmark dimensions:

- overall accuracy including abstentions
- selective accuracy and coverage
- ECE, Brier score and NLL
- risk-coverage curve
- p50/p95/p99 latency under a declared concurrency
- total and per-1k cost
- irrelevant-context, prompt-injection and option-order robustness
- high-risk slices and OOD cases
- provider error/timeout/fallback rates

Use a pinned Jev version after tuning; Pydantic's TypeSafe documentation explicitly recommends pinning because `jev-latest` can move and shift confidence distributions. The public example is `typesafe:jev-1.13.0` as of September 2026.

The bundled `demo.jsonl` is synthetic smoke data only. It cannot support a general "better than Jev" claim. Use real, human-labeled traces and preserve dataset hashes and provider versions for any public result.


## External-baseline validity

A provider outage, DNS failure, authentication failure, timeout, or circuit-open result is **not** a wrong Jev prediction. The paired comparison is invalid unless every frozen Jev row returns a real typed decision and the API reports an observed model id. For versioned runs, the observed id must match the requested pinned model. `benchmarks/gate.py` fails closed otherwise.

Benchmark latency is measured around the entire engine call. Benchmark cost uses `total_cost_usd` from the cascade where available, so retries/fallback calls cannot disappear from the accounting.

The dependency-free `DirectJevProvider` uses the documented `POST /v1/systemone` contract and reads `TYPESAFE_API_KEY` only from process memory/environment unless explicitly provided by the caller. Never commit credentials or put them in benchmark JSON.

## Recommended tracks

Report at least two tracks instead of collapsing everything into one headline score:

1. **Semantic-only:** decisions where both systems must make the same semantic judgment. This measures model/provider quality directly.
2. **Harness-mixed:** representative production traffic including deterministic policy/rules, semantic routing, OOD/abstention and model fallback. This measures the end-to-end system Agent Reflex is designed to optimize.

A mixed-workload speed/cost advantage does not imply the local semantic scorer is intrinsically smarter than Jev. Keep those claims separate.
