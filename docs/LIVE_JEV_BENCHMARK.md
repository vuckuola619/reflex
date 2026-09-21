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

A subsequent authorized live eight-case smoke succeeded against observed model `jev-1.13.0`: Jev answered 8/8 correctly; the then-current local demo answered 6/8 and abstained twice. This is a tiny synthetic smoke, not superiority evidence. Results are stored locally in `artifacts/live-jev.json`.

The newer real-observation comparison is `benchmarks/real_workload.py`. After approval, all 42 live calls succeeded against observed `jev-1.13.0`. Jev and Reflex exact checks both got 17/17 primary observations correct. See the [live comparison](JEV_COMPARISON.md) for latency, costs, semantic results, stress cases and limits. Never score an unavailable/unauthorized remote run as model error.
