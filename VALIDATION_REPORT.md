# CATENA v6.1.0 validation report

Validation date: 2026-07-26

## Scope

This report covers the clean v6.1 source tree and the post-E01 scientific revision. It does **not** claim that the official KDA/GDN2 CUDA backend or the official KVEraser integration has run on the target Blackwell server.

## Passed checks

- Python `compileall`: PASS
- Pytest: PASS, 20 test cases
- Experiment/config correspondence: PASS, 10 experiment entry points and 10 YAML configs
- CPU dry-run sequence E00, E01, E01b, E02-E08: PASS
- E07 Transformer boundary: correctly blocked as `MOCK_ONLY`
- Research-plan XeLaTeX/Biber clean build: PASS
- Research-plan render inspection: PASS, 9/9 pages
- Frozen E00/E01 dependency list documented for the post-E01 patch

## Scientific changes verified by tests or dry runs

- box-constrained tied/dual reachability
- equal-weight correction/retention behavioral readout
- operation-fixed-effect unseen-geometry calibration path
- same-parameter two-output tied/dual controller
- H2 raw metrics, SESOI/equivalence/noninferiority claim gate
- commuting/common-rotated/noncommuting operator-family generation
- multi-restart shared-basis joint diagonalization
- eight-seed functional-intervention execution path
- semantic split and stratified audit export
- retrieve-once cached-snapshot cost path
- evidence-based claim freeze

## Not executed in this build environment

- target-server E00/E01 GPU runs
- official accelerated KDA/GDN2 numerical parity
- official pretrained recurrent-LM transfer
- official KVEraser training/evaluation
- `ruff` and `mypy` (not installed in this build environment)

Mock/reference outputs remain blocked from scientific claim gates.
