# Live Jev benchmark

Agent Reflex compares against the TypeSafe System One HTTP API directly. No SDK is required.

```bash
export TYPESAFE_API_KEY='your-key'
export JEV_MODEL='jev-1.13.0'
./scripts/benchmark_jev.sh path/to/frozen.jsonl benchmark-jev.json
```

The key is read from the process environment and is never included in the benchmark JSON.

A valid report requires:

- every Jev holdout call returned a typed decision;
- Jev coverage is 100% for the baseline track;
- the response contains an observed model id;
- a pinned requested model exactly matches the observed model;
- the same frozen dataset hash is used for Agent Reflex and Jev.

Network, authentication, quota and provider failures invalidate the comparison. They must never be interpreted as Jev accuracy failures.

The sandbox used to prepare RC2 blocks outbound network access. A live attempt on 2026-09-20 therefore failed closed with `jev_comparison_valid=false`; this is an environment limitation, not a Jev result.
