# Agent Reflex 1.0.0rc2 — QC Report

Prepared: 2026-09-20

## Offline validation completed

- Python compile check: PASS
- Unit/integration tests: **38/38 PASS**
- Branch-aware core coverage: **84.68%** (CI floor: 80%) (CI floor: 80%)
- REST authentication integration: PASS
- High-risk fail-closed: PASS
- OOD rejection: PASS
- NaN/invalid probability typed firewall: PASS
- Timeout/retry/circuit breaker paths: PASS
- Cost/call/deadline budget paths: PASS
- Secret-redaction telemetry test: PASS
- Telemetry-failure isolation: PASS
- Async bounded batching: PASS
- Benchmark train/validation/test smoke: PASS
- Benchmark provenance hash/platform recording: PASS
- YAML/TOML/JSON configuration parse: PASS
- 10k local load smoke at concurrency 64: ~18.3k decisions/sec for the lightweight rules+prototype cascade in this workspace

## Synthetic benchmark snapshot

The bundled 8-case test split is intentionally too small to support a superiority claim.

- coverage: 0.75
- selective accuracy: 1.00
- overall accuracy: 0.75
- ECE: ~0.0362
- Brier: ~0.0944
- cost: $0 for local rules/prototype path
- option-order stability: 1.00

Abstention is counted against overall accuracy; it is not hidden as a success.

## External gates intentionally not fabricated

1. **Live Jev comparison** — the user supplied a key and the direct HTTP path was attempted on 2026-09-20, but this sandbox has no outbound network/DNS. The resulting run is marked `jev_comparison_valid=false` and the gate rejects it. The key was not written to the repository or benchmark output. Run the same command in connected CI/local infrastructure to obtain valid evidence.
2. **Dependency lock/SBOM generation** — this execution environment could not resolve PyPI DNS. `scripts/lock.sh` generates `uv.lock`, standards-based `pylock.toml`, and CycloneDX once run in a connected trusted environment.
3. **Target-infrastructure soak/chaos testing** — requires the actual deployment topology/providers.
4. **Independent security review** — required before granting a concrete agent destructive/external side effects.

## Browser-verified production references

- MCP 2026-07-28 introduced a stateless protocol core; Python SDK v2 is the stable line and supports that revision.
  - https://blog.modelcontextprotocol.io/posts/2026-07-28/
  - https://github.com/modelcontextprotocol/python-sdk
- OpenTelemetry Python traces and metrics are stable; library instrumentation should depend on the API and let applications configure SDK/exporters.
  - https://opentelemetry.io/docs/languages/python/
- OPA supports REST policy decisions, bundles, decision logs, health/status, and sidecar-style distributed enforcement. OPA v1.19.0 was the latest release checked during this build.
  - https://www.openpolicyagent.org/docs
  - https://www.openpolicyagent.org/docs/management-bundles
  - https://www.openpolicyagent.org/docs/management-decision-logs
- vLLM supports constrained structured outputs through choice/regex/JSON/grammar.
  - https://docs.vllm.ai/en/latest/features/structured_outputs/
- `pylock.toml` is the standardized reproducible environment format defined from PEP 751.
  - https://packaging.python.org/en/latest/specifications/pylock-toml/
- PyPI recommends OIDC Trusted Publishing from supported CI systems.
  - https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- Sigstore Python supports keyless signing and verification.
  - https://docs.sigstore.dev/language_clients/python/
- Pydantic AI's current TypeSafe/Jev docs recommend pinning a version after threshold tuning; `jev-1.13.0` is a documented versioned example.
  - https://pydantic.dev/docs/ai/models/typesafe/


## RC2 benchmark-integrity fixes

- Added dependency-free `DirectJevProvider` for the official System One HTTP contract.
- External provider failures can no longer masquerade as baseline errors/low accuracy.
- Comparison requires 100% successful Jev holdout coverage and a returned model id.
- Pinned version mismatch invalidates the run.
- Benchmark latency now uses full wall-clock engine time.
- Benchmark cost now includes total cascade cost where available.
- Live-attempt output contains no API key.
