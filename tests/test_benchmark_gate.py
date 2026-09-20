from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_gate(tmp_path: Path, report: dict) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    gate = Path(__file__).parents[1] / "benchmarks" / "gate.py"
    return subprocess.run([sys.executable, str(gate), str(path)], capture_output=True, text=True)


def test_gate_rejects_failed_external_baseline(tmp_path):
    result = _run_gate(tmp_path, {"jev": {"coverage": 0.0}, "jev_comparison_valid": False, "jev_comparison_errors": ["timeout"]})
    assert result.returncode != 0
    assert "INVALID JEV COMPARISON" in result.stderr


def test_gate_rejects_missing_observed_model(tmp_path):
    result = _run_gate(tmp_path, {
        "jev_comparison_valid": True,
        "provenance": {"jev_observed_models": []},
        "agent_reflex": {},
        "jev": {"coverage": 1.0},
    })
    assert result.returncode != 0
    assert "observed Jev model" in result.stderr

def test_gate_rejects_pinned_model_mismatch(tmp_path):
    result = _run_gate(tmp_path, {
        "jev_comparison_valid": True,
        "provenance": {"jev_requested_model": "jev-1.13.0", "jev_observed_models": ["jev-1.14.0"]},
        "agent_reflex": {},
        "jev": {"coverage": 1.0},
    })
    assert result.returncode != 0
    assert "requested jev-1.13.0" in result.stderr
