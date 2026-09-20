# Contributing

Use Python 3.11+ and create an isolated environment.

```bash
python -m pip install -e '.[dev]'
ruff check src tests benchmarks examples
pytest --cov=agent_reflex
PYTHONPATH=src python benchmarks/iterate.py --json-out benchmark.json
python -m build
python -m twine check dist/*
```

Provider changes must include failure-path tests. Threshold/calibration changes must be fit on validation data only. Do not claim superiority over another runtime unless the same frozen holdout, provider versions, hardware class, concurrency, warm/cold policy, and cost accounting were used.
