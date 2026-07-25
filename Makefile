SHELL := /bin/bash
CONDA_ENV ?= catena
CONDA_RUN ?= conda run --no-capture-output -n $(CONDA_ENV)
PYTHON ?= $(CONDA_RUN) python

.PHONY: help install audit config-audit smoke test runtime data h1h2 teacher h3-sweep h3-main h3-eval transformer h4 h4-eval profile toolcalls freeze bundle lint clean

help:
	@echo "All Python targets use the existing Conda environment: $(CONDA_ENV)"
	@echo "make audit          Hardware/software audit"
	@echo "make config-audit   Validate YAML and shell references"
	@echo "make runtime        E01 four-GPU runtime gates"
	@echo "make data           E02 fixed dataset generation"
	@echo "make h1h2           E03-E04 diagnostic experiments"
	@echo "make teacher        E05 RWKV teacher cache"
	@echo "make h3-sweep       E06 K=4/8/16 + generic sweep"
	@echo "make h3-main        E07 three seeds and ablations"
	@echo "make h3-eval        E07 main/stress evaluation"
	@echo "make transformer    E08 Transformer boundary"
	@echo "make h4             E09 composition/control training"
	@echo "make h4-eval        E09 six-checkpoint evaluation"
	@echo "make profile        E10 system profile"
	@echo "make toolcalls      E11 non-learned tool-call baselines"
	@echo "make freeze         E12 clean environment/config freeze"

install:
	@echo "ERROR: CATENA does not install or modify environments from Makefile." >&2
	@echo "Use the existing Conda environment '$(CONDA_ENV)' and run 'make audit'." >&2
	@exit 2

audit:
	bash scripts/00_bootstrap_and_audit.sh

config-audit:
	$(PYTHON) -m catena.cli config-audit

smoke:
	$(PYTHON) -m catena.cli smoke

test:
	PYTHONPATH=src $(PYTHON) -m pytest

runtime:
	bash scripts/01_runtime_gates.sh

data:
	bash scripts/02_generate_and_validate_data.sh

h1h2:
	bash scripts/03_h1_h2_pilot_4gpu.sh

teacher:
	bash scripts/04_build_rwkv_teacher_4gpu.sh

h3-sweep:
	bash scripts/05_train_h3_4gpu.sh

h3-main:
	bash scripts/06_train_h3_ablations_4gpu.sh

h3-eval:
	bash scripts/06_eval_h3_4gpu.sh

transformer:
	bash scripts/07_transformer_boundary_4gpu.sh

h4:
	bash scripts/08_h4_composition_4gpu.sh

h4-eval:
	bash scripts/08_eval_h4_4gpu.sh

profile:
	bash scripts/09_profile_4gpu.sh

toolcalls:
	bash scripts/10_naturalized_toolcalls_4gpu.sh

freeze:
	bash scripts/11_clean_rerun_and_freeze.sh

bundle:
	bash scripts/12_bundle_results.sh

lint:
	$(CONDA_RUN) ruff check src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
