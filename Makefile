.PHONY: test compile dry lint typecheck

test:
	python -m pytest -q

compile:
	python -m compileall -q src experiments tests

dry:
	python tools/run_all_dry.py --device cpu

lint:
	ruff check src experiments tests tools

typecheck:
	mypy src
