# CATENA v6.2 Post-Core File Map

## Entry point와 config

| Stage | Python | YAML | Protocol doc |
|---|---|---|---|
| E10 | `experiments/e10_learned_rank_scaling.py` | `configs/e10_learned_rank_scaling.yaml` | `docs/E10_LEARNED_RANK_SCALING_KO.md` |
| E11 | `experiments/e11_representation_control_coadaptation.py` | `configs/e11_representation_control_coadaptation.yaml` | `docs/E11_REPRESENTATION_COADAPTATION_KO.md` |
| E12 | `experiments/e12_control_algebra_lattice.py` | `configs/e12_control_algebra_lattice.yaml` | `docs/E12_CONTROL_ALGEBRA_LATTICE_KO.md` |
| E13a pilot (immutable) | `experiments/e13a_sequence_floor_throughput.py` | `configs/e13a_sequence_floor_throughput.yaml` | `docs/E13_TRANSACTIONAL_SEQUENCE_MEMORY_KO.md` |
| E13a-R1 | `experiments/e13a_r1_sequence_floor_throughput.py` | `configs/e13a_r1_sequence_floor_throughput.yaml` | `docs/E13A_R1_SEQUENCE_CALIBRATION_KO.md` |
| E13b | `experiments/e13b_transactional_sequence_memory.py` | `configs/e13b_transactional_sequence_memory.yaml` | 동일 |
| E13c | `experiments/e13c_transactional_sequence_aggregate.py` | `configs/e13c_transactional_sequence_aggregate.yaml` | 동일 |
| E14 | `experiments/e14_plan_continuation.py` | `configs/e14_plan_continuation.yaml` | `docs/E14_PLAN_CONTINUATION_KO.md` |
| E15 | `experiments/e15_official_backend_gate.py` | `configs/e15_official_backend_gate.yaml` | `docs/E15_OFFICIAL_BACKEND_GATE_KO.md` |
| E16 | `experiments/e16_core_evidence_freeze.py` | `configs/e16_core_evidence_freeze.yaml` | `docs/E16_EVIDENCE_FREEZE_KO.md` |

## 공통 라이브러리

- `src/catena/data/learned_rank.py`
- `src/catena/data/representation_dynamics.py`
- `src/catena/data/control_lattice.py`
- `src/catena/data/transactional_sequence.py`
- `src/catena/models/operator_controllers.py`
- `src/catena/models/coadaptation.py`
- `src/catena/models/lattice_controllers.py`
- `src/catena/models/sequence_memory.py`
- `src/catena/training/postcore.py`
- `src/catena/training/lattice_training.py`
- `src/catena/training/sequence_training.py`
- `src/catena/eval/postcore_metrics.py`
- `src/catena/eval/evidence_freeze.py`

## 운용 도구

- `scripts/launch_postcore_wave1.sh`
- `scripts/launch_sequence_wave.sh`
- `scripts/launch_sequence_if_go.sh`
- `scripts/check_postcore_status.sh`
- `scripts/run_postcore_dry.sh`
- `tools/postcore_status.py`
- `tools/run_postcore_dry.py`

## Codex와 문서

- `CODEX_POSTCORE.md`
- `docs/NEXT_ACTIONS_KO.md`
- `docs/POSTCORE_DEPENDENCY_DAG_KO.md`
- `docs/POSTCORE_ARTIFACT_CONTRACT_KO.md`
- `docs/POSTCORE_4GPU_SCHEDULE_KO.md`
- `docs/POSTCORE_RESULT_INTERPRETATION_KO.md`
- `docs/POSTCORE_CLAIM_GATES_KO.md`
- `docs/CODEX_POSTCORE_TASKS.md`

## 연구계획

- `papers/postcore_research_plan/CATENA_control_algebra_research_plan_v6.2_ko.pdf`
- `papers/postcore_research_plan/main.tex`
- `papers/postcore_research_plan/references.bib`
