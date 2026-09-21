"""Real loopback socket fault injection. No external service or credential needed."""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_reflex import DecisionBudget, DecisionRequest, ProviderStage, Reflex
from agent_reflex.engine import CircuitBreaker
from agent_reflex.providers.jev_http import DirectJevProvider


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


class Handler(BaseHTTPRequestHandler):
    mode = "healthy"
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).calls += 1
        mode = type(self).mode
        if mode == "stall":
            time.sleep(0.3)
        code = 503 if mode == "outage" else 200
        body = b"broken JSON" if mode == "invalid" else json.dumps({
            "model": "fixture-model", "answers": {"answer": {"noul": 0.95}},
            "usage": {"input_tokens": 1},
        }).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


async def run(url):
    request = DecisionRequest({"exit_code": 0}, "Did the command succeed?")
    results = []
    for mode in ("healthy", "outage", "invalid", "stall"):
        Handler.mode = mode
        engine = Reflex.local()
        engine.default_budget = DecisionBudget(deadline_ms=250, max_cost_usd=0.01, max_provider_calls=2)
        engine.stages.insert(0, ProviderStage(DirectJevProvider(api_key="local-fixture", base_url=url, timeout_s=0.1), timeout_ms=40))
        start = time.perf_counter()
        decision = await engine.adecide(request)
        elapsed = (time.perf_counter() - start) * 1000
        assert decision.value is True and not decision.abstained, mode
        assert decision.provider == ("jev-http" if mode == "healthy" else "evidence"), mode
        assert elapsed < 250, (mode, elapsed)
        results.append({"fault": mode, "pass": True, "latency_ms": elapsed,
                        "provider": decision.provider, "attempts": decision.metadata["attempts"]})
        await engine.aclose()
    Handler.mode = "outage"
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.15)
    engine = Reflex.local()
    engine.default_budget = DecisionBudget(deadline_ms=250, max_cost_usd=0.01, max_provider_calls=2)
    engine.stages.insert(0, ProviderStage(DirectJevProvider(api_key="local-fixture", base_url=url, timeout_s=0.1), breaker=breaker))
    initial = Handler.calls
    for _ in range(3):
        assert (await engine.adecide(request)).provider == "evidence"
    assert Handler.calls - initial == 2
    await asyncio.sleep(0.16)
    Handler.mode = "healthy"
    assert (await engine.adecide(request)).provider == "jev-http"
    results.append({"fault": "circuit_open_then_recovery", "pass": True, "failed_network_calls": 2})
    await engine.aclose()
    Path("artifacts/socket-fire-drill.json").write_text(json.dumps({"scope": "real loopback HTTP; controlled fixtures, not actual Jev outage", "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            asyncio.run(run(f"http://127.0.0.1:{server.server_port}"))
        finally:
            server.shutdown()
            thread.join(timeout=2)
