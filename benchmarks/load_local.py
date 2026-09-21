from __future__ import annotations

import argparse
import asyncio
from statistics import mean
from time import perf_counter

from agent_reflex import DecisionKind, DecisionRequest, ProviderStage, ReflexEngine
from agent_reflex.providers import Prototype, PrototypeProvider, Rule, RuleProvider


def build_engine() -> ReflexEngine:
    rules = RuleProvider(
        [
            Rule("git-read", lambda r: str(r.state).startswith(("git status", "git diff")), "allow"),
            Rule("root-delete", lambda r: "rm -rf /" in str(r.state), "deny"),
        ]
    )
    proto = PrototypeProvider(
        [
            Prototype("changed authentication session token", "auth"),
            Prototype("dependency lockfile package version", "full"),
            Prototype("small parser utility helper", "unit"),
        ]
    )
    return ReflexEngine(
        [
            ProviderStage(rules, threshold=1.0, timeout_ms=20, sanitize_context=False),
            ProviderStage(proto, threshold=0.55, timeout_ms=30, max_ood=0.9),
        ]
    )


async def run(n: int, concurrency: int) -> None:
    engine = build_engine()
    templates = [
        DecisionRequest("git status --short", "action?", DecisionKind.CHOICE, ["allow", "deny"]),
        DecisionRequest("changed authentication session", "tests?", DecisionKind.CHOICE, ["auth", "unit", "full"]),
        DecisionRequest("dependency lockfile updated", "tests?", DecisionKind.CHOICE, ["auth", "unit", "full"]),
        DecisionRequest("small parser helper", "tests?", DecisionKind.CHOICE, ["auth", "unit", "full"]),
    ]
    reqs = [templates[i % len(templates)] for i in range(n)]
    started = perf_counter()
    out = await engine.adecide_many(reqs, concurrency=concurrency)
    elapsed = perf_counter() - started
    latencies = [x.latency_ms for x in out]
    print(
        {
            "n": n,
            "concurrency": concurrency,
            "elapsed_s": elapsed,
            "decisions_per_second": n / elapsed,
            "mean_provider_latency_ms": mean(latencies),
            "abstention_rate": sum(x.abstained for x in out) / n,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--concurrency", type=int, default=64)
    a = p.parse_args()
    asyncio.run(run(a.n, a.concurrency))
