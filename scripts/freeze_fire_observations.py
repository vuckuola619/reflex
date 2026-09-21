"""Freeze actual subprocess and loopback API observations before comparison."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, UTC
import httpx2 as httpx

root = Path(__file__).resolve().parents[1]
os.chdir(root)
folder = root / "artifacts" / "fire-live"
folder.mkdir(parents=True, exist_ok=True)
path = folder / "observations.jsonl"
if path.exists():
    raise SystemExit("Frozen observations already exist; use a new explicitly named evaluation instead of overwriting.")
commands = [
    ["-m", "pytest", "tests/test_core.py", "-q"],
    ["-m", "pytest", "tests/test_production.py", "-q"],
    ["-m", "pytest", "tests/test_jev_http.py", "-q"],
    ["-m", "pytest", "tests/test_local_regressions.py", "-q"],
    ["-m", "pytest", "tests/test_service_regressions.py", "-q"],
    ["-m", "pytest", "tests/test_fire_drill.py", "-q"],
    ["-m", "ruff", "check", "src"],
    ["-m", "ruff", "format", "--check", "src"],
    ["-m", "compileall", "-q", "src"],
    ["-c", "from agent_reflex import Reflex; assert callable(Reflex.local)"],
    ["-c", 'from agent_reflex import DecisionKind; assert DecisionKind("route")'],
    ["-c", 'from agent_reflex import DecisionKind; assert DecisionKind("policy")'],
]
rows = []
child_env = dict(os.environ, PYTHONPATH=str(root / "src"))
for i, args in enumerate(commands):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=120, env=child_env)
    evidence = (r.stdout + "\n" + r.stderr).replace(str(root), "<repository>")[-4500:]
    rows.append(
        {
            "id": f"command-{i:02}",
            "kind": "boolean",
            "question": "Did this specific verification command pass? A nonzero exit status means the gate did not pass, even if some individual checks succeeded.",
            "state": {"command": ["python", *args], "exit_code": r.returncode, "output": evidence},
            "gold": r.returncode == 0,
            "source": "real repository subprocess",
            "date_utc": datetime.now(UTC).isoformat(),
        }
    )
with httpx.Client(base_url="http://127.0.0.1:8080", timeout=5) as c:
    for url in ["/health", "/ready", "/v1/decisions/choice", "/v1/decisions/route", "/this-endpoint-does-not-exist"]:
        r = (
            c.get(url)
            if url in ["/health", "/ready", "/this-endpoint-does-not-exist"]
            else c.post(url, json={"state": "x", "question": "q", "options": ["a", "b"]})
        )
        rows.append(
            {
                "id": f"http-{len(rows):02}",
                "kind": "boolean",
                "question": "Did this probe return a successful HTTP 2xx response?",
                "state": {"endpoint": url, "status_code": r.status_code, "response": r.text[:2000]},
                "gold": 200 <= r.status_code < 300,
                "source": "real loopback HTTP probe",
                "date_utc": datetime.now(UTC).isoformat(),
            }
        )
raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
path.write_bytes(raw)
manifest = {
    "sha256": hashlib.sha256(raw).hexdigest(),
    "n": len(rows),
    "positive": sum(r["gold"] for r in rows),
    "negative": sum(not r["gold"] for r in rows),
    "scope": "Observed repository gates and real loopback probes; convenience sample, not independent production workload. Fault-injection test output included and labeled by command.",
    "frozen_utc": datetime.now(UTC).isoformat(),
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
