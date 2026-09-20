# Roadmap

## Promotion gate: 1.0.0rc1 -> 1.0.0

The codebase already includes the production runtime controls. Promotion is evidence-driven, not feature-driven.

- [ ] Commit a generated `uv.lock` and `pylock.toml` from trusted connected CI.
- [ ] Build a real human-labeled benchmark (minimum target: thousands of representative decisions, with high-risk/OOD slices).
- [ ] Run paired benchmark against pinned `typesafe:jev-1.13.0` or the deliberately selected newer pinned release.
- [ ] Pass benchmark evidence gate without tuning on final holdout.
- [ ] Run load + soak + provider-outage chaos tests on target infrastructure.
- [ ] Validate OPA policy bundles and decision-log pipeline in staging.
- [ ] Exercise OpenTelemetry export against the target collector/backend.
- [ ] Complete threat model / independent security review for enabled agent actions.
- [ ] Shadow -> 1% -> 10% -> 50% -> 100% rollout with automatic rollback thresholds.

## Post-1.0 candidates

- Pluggable isotonic calibration and conformal abstention.
- Native semantic-router adapter and threshold optimizer.
- vLLM structured-output/logprob provider once cross-model confidence extraction is stable enough to expose a consistent contract.
- Persisted feedback/eval store adapters for Postgres/ClickHouse.
- Signed Agent Skills manifests and policy-bundle provenance.
