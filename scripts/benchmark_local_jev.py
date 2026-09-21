"""Load the user-provided local key into process memory, then run the benchmark."""

import os
import runpy
import sys
from pathlib import Path

from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
os.chdir(root)
values = dotenv_values(root / ".env")
key = os.environ.get("TYPESAFE_API_KEY") or values.get("TYPESAFE_API_KEY") or values.get("jev_key")
if not key:
    raise SystemExit("Set TYPESAFE_API_KEY or provide it in .env")
os.environ["TYPESAFE_API_KEY"] = key
sys.argv = [
    "benchmarks/iterate.py",
    "--compare-jev",
    "--jev-model",
    "jev-1.13.0",
    "--json-out",
    "artifacts/live-jev.json",
]
runpy.run_path(root / "benchmarks/iterate.py", run_name="__main__")
