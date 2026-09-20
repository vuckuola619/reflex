.PHONY: test lint benchmark lock build verify

test:
	PYTHONPATH=src python -m pytest -q

lint:
	ruff check src tests benchmarks examples

benchmark:
	PYTHONPATH=src python benchmarks/iterate.py --json-out benchmark-local.json

lock:
	./scripts/lock.sh

build:
	python -m build
	python -m twine check dist/*

verify:
	./scripts/verify.sh
