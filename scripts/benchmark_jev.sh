#!/usr/bin/env bash
set -euo pipefail

: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY in the environment; never put it in this script or the repo.}"
DATASET="${1:-benchmarks/demo.jsonl}"
OUT="${2:-benchmark-jev.json}"
MODEL="${JEV_MODEL:-jev-1.13.0}"

PYTHONPATH=src python benchmarks/iterate.py \
  --dataset "$DATASET" \
  --compare-jev \
  --jev-adapter http \
  --jev-model "$MODEL" \
  --json-out "$OUT"

PYTHONPATH=src python benchmarks/gate.py "$OUT"
