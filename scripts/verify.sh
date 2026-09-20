#!/usr/bin/env sh
set -eu
python -m compileall -q src tests benchmarks examples
python -m pytest --cov=agent_reflex --cov-report=term-missing
ruff check src tests benchmarks examples
python -m build
python -m twine check dist/*
PYTHONPATH=src python benchmarks/iterate.py --json-out benchmark-local.json
