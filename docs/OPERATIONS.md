# Operations

## Serving

REST is authenticated by default. Set `AGENT_REFLEX_API_KEYS` and run:

```bash
agent-reflex serve --factory examples.server_app:build_engine --host 0.0.0.0 --port 8080
```

MCP uses the v2 Python SDK and Streamable HTTP. The 2026-07-28 protocol is sessionless by design; `stateless_http=True` also avoids legacy sticky-session requirements.

```bash
agent-reflex mcp --factory examples.server_app:build_engine
```

## Policy

Run OPA as a sidecar for authorization/hard policy. OPA bundles are the preferred policy distribution mechanism for many-instance deployments, and OPA decision logs can be collected for audit/offline debugging. The example pins OPA `1.19.0-static`.

## Observability

The core includes JSONL and OpenTelemetry sinks. Recommended alerts: p95/p99 latency, abstention/fallback rate, circuit-open count, provider errors/timeouts, OOD rate, high-risk decisions, policy denials, and cost/1k.

## Release and supply chain

- Generate `uv.lock`, standardized `pylock.toml`, and optional CycloneDX SBOM with `scripts/lock.sh` in an internet-connected environment.
- CI runs tests/coverage, lint, build checks, pip-audit and CodeQL.
- PyPI release uses OIDC Trusted Publishing rather than a long-lived upload token.
- Release artifacts are signed with Sigstore keyless signing.

The execution environment used to build this repository had no outbound PyPI DNS, so no lockfile was fabricated. Generate and commit the lock from a connected trusted CI/dev environment before a production deployment.
