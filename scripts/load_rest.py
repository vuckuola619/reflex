"""Short loopback REST load smoke; not a production capacity claim."""

import asyncio
import json
import platform
from math import ceil
from pathlib import Path
from time import perf_counter

import httpx2 as httpx
import psutil


async def main():
    output = []
    for rate in [1, 10, 100]:
        n = rate * 5
        lat = []
        errors = 0
        active = 0
        max_active = 0
        process = psutil.Process(int(Path("artifacts/server.pid").read_text()))
        children = process.children(recursive=True)
        if children:
            process = children[-1]
        cpu_start = process.cpu_times()
        cpu_before = cpu_start.user + cpu_start.system
        started = perf_counter()
        async with httpx.AsyncClient(timeout=5) as client:

            async def one(i, start_time=started, target_rate=rate, latencies=lat):
                nonlocal errors, active, max_active
                due = start_time + i / target_rate
                await asyncio.sleep(max(0, due - perf_counter()))
                active += 1
                max_active = max(active, max_active)
                try:
                    r = await client.post(
                        "http://127.0.0.1:8080/v1/decide",
                        json={
                            "state": "git status --short",
                            "question": "action?",
                            "kind": "choice",
                            "options": ["allow", "deny"],
                            "risk": 0.1,
                        },
                    )
                    if r.status_code != 200 or r.json().get("value") != "allow":
                        errors += 1
                except Exception:
                    errors += 1
                finally:
                    latencies.append((perf_counter() - due) * 1000)
                    active -= 1

            await asyncio.gather(*(one(i) for i in range(n)))
        elapsed = perf_counter() - started
        cpu = process.cpu_times()
        lat.sort()
        output.append(
            {
                "target_rps": rate,
                "samples": n,
                "duration_s": elapsed,
                "achieved_rps": n / max(elapsed, 5),
                "p50_ms": lat[ceil(0.50 * n) - 1],
                "p95_ms": lat[ceil(0.95 * n) - 1],
                "p99_ms": lat[ceil(0.99 * n) - 1],
                "error_rate": errors / n,
                "server_cpu_core_percent": 100 * (cpu.user + cpu.system - cpu_before) / elapsed,
                "server_rss_bytes": process.memory_info().rss,
                "max_client_inflight": max_active,
                "server_queue_depth": "not instrumented",
                "cost_usd": 0,
            }
        )
    report = {
        "track": "loopback REST rules-only smoke",
        "platform": platform.platform(),
        "gpu": "not used",
        "limitations": "5-second tracks; tiny samples; no remote models; not a sustained production capacity test",
        "results": output,
    }
    Path("artifacts/rest-load.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


asyncio.run(main())
