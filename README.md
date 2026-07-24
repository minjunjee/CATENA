# CATENA: Transaction-Conditioned State Transport

REALM @ EMNLP 2026 제출을 위한 **4-GPU full-access baseline repository**다. 중심 질문은 다음 하나다.

> 검증된 외부 메모리 transaction이 발생했을 때, 이미 형성된 recurrent execution state를 짧은 native forward로 갱신해 최신 canonical state의 exact-refresh behavior와 일치시킬 수 있는가?

Transformer는 메인 방법이 아니라 경계와 공정성을 확인하는 비교군이다. KV append뿐 아니라 reset/capsule, 실제 prefix KV를 재사용하는 oracle suffix re-prefill, full re-prefill, parameter-matched learned soft patch를 비교한다.

## 대상 서버

- GPU: NVIDIA RTX PRO 6000 Blackwell 계열 4장; 실제 제품명/VRAM/MIG/P2P는 audit로 확정
- NVIDIA driver: `580.126.16`
- System CUDA toolkit: `13.0`
- Python: `3.11`
- PyTorch pin: `2.12.1+cu130`
- Precision: BF16, metric accumulation FP32

3B 모델은 기본적으로 GPU 한 장당 한 process로 실행한다. 네 GPU를 한 DDP job에 묶기보다 teacher shards, seed, ablation, Transformer boundary를 네 개의 독립 experiment lane으로 병렬화한다.

## 첫 실행

```bash
cd CATENA_REALM2026_baseline_repo_v0.6.0
bash scripts/00_bootstrap_and_audit.sh
source .venv/bin/activate
source scripts/setup_paths.sh
bash scripts/install_rwkv_fla.sh
bash scripts/01_runtime_gates.sh
bash scripts/02_generate_and_validate_data.sh
```

장시간 작업은 SSH와 분리한다.

```bash
bash scripts/run_in_tmux.sh catena-h1h2 "bash scripts/03_h1_h2_pilot_4gpu.sh"
tmux attach -t catena-h1h2
```

## 시간 순서의 실험

| 단계 | 질문 | 핵심 산출물 |
|---|---|---|
| E00 | 서버와 CUDA/PyTorch가 재현 가능하게 고정됐는가 | host/topology/environment manifest |
| E01 | RWKV recurrent cache와 Qwen KV cache adapter가 올바른가 | parity/gradient/runtime gate |
| E02 | 데이터가 coherence와 leakage를 분리하는가 | UpdateBench fixed splits |
| E03/H1 | stale execution state가 실제 오류를 만드는가 | stale-exact paired gap |
| E04/H2 | 어떤 transaction 표현이 필요하며 누출은 아닌가 | plain/typed/closure/reset/retrieval ablation |
| E05 | exact teacher를 학습에 재사용할 수 있는가 | candidate-distribution cache |
| E06 | learned transport가 학습되며 적절한 slot 수는 무엇인가 | 300-step gate + K=4/8/16 sweep |
| E07/H3 | CATENA가 text/reset/retrieval/generic slot보다 Pareto를 개선하는가 | 3-seed main result and ablations |
| E08 | Transformer cache repair와 비교한 경계는 어디인가 | append/capsule/suffix/full/soft-patch regime map |
| E09/H4 | composition loss가 긴 chain drift를 줄이는가 | length 4/8/16 drift comparison |
| E10 | 실제 서버 비용은 무엇인가 | p50/p95 latency, TTFA, decode, state/KV bytes |
| E11 | candidate coherence가 실제 tool action으로 이어지는가 | JSON schema and simulator success |
| E12 | 어떤 claim까지 재현 가능한가 | clean rerun, figures, anonymous package |

세부 데이터, 모델, loss, metric과 gate는 [`docs/EXPERIMENT_RUNBOOK_KO.md`](docs/EXPERIMENT_RUNBOOK_KO.md)에 있다.

## 저장소 문서

- [`docs/START_HERE_KO.md`](docs/START_HERE_KO.md): 서버 접속 후 실행 순서와 네 GPU lane
- [`docs/EXPERIMENT_RUNBOOK_KO.md`](docs/EXPERIMENT_RUNBOOK_KO.md): 모든 실험의 시간 순서, 데이터, 학습, loss, metric, 성공/반증 조건
- [`docs/EXPERIMENT_MATRIX_KO.md`](docs/EXPERIMENT_MATRIX_KO.md): E00-E12의 모델·데이터·loss·metric·gate 한눈표
- [`docs/REPO_GUIDE_KO.md`](docs/REPO_GUIDE_KO.md): stage 기준 폴더와 산출물 흐름
- [`docs/CLAIM_GATES_KO.md`](docs/CLAIM_GATES_KO.md): 결과별 허용 주장
- [`docs/SERVER_SETUP_KO.md`](docs/SERVER_SETUP_KO.md): CUDA 13/4-GPU 설치와 운영
- [`docs/BACKEND_STATUS_KO.md`](docs/BACKEND_STATUS_KO.md): differentiable RWKV adapter의 gate와 한계
- [`BASELINE_STATUS.md`](BASELINE_STATUS.md): 현재 검증된 부분과 서버에서 확인할 부분

## 주요 명령

```bash
python -m catena.cli experiment-list
python -m catena.cli data-generate --config configs/data/pilot.yaml
python -m catena.cli eval-inference --config configs/experiments/e03_h1.yaml
python -m catena.cli teacher-cache --config configs/experiments/e05_rwkv_teacher.yaml --split train
python -m catena.cli train-h3 --config configs/experiments/e06_h3_slots8.yaml --seed 11 --max-steps 300
python -m catena.cli train-h4 --config configs/experiments/e09_h4_train.yaml --seed 11
python -m catena.cli profile-system --config configs/experiments/e10_profile.yaml --model-index 0
```

## Backbone 고정 원칙

메인 RWKV 결과는 `fla-hub/rwkv7-2.9B-g1`과 그 원본인 `rwkv7-g1-2.9b-20250519-ctx4096.pth`의 동일 weight 계열로 고정한다. 공식 PTH runtime은 같은 source checkpoint의 교차 검증에만 사용한다. 최신 PTH와 오래된 FLA conversion을 한 결과표에 섞지 않는다.

## 현재 구현 상태

CPU/mock pipeline과 unit tests는 동작한다. H1/H2의 text/inference policy, Qwen KV cache policy, teacher materialization, H3/H4 trainer, profiling과 tool-call runner가 포함돼 있다.

H3/H4의 실제 과학적 run은 대상 서버에서 pinned FLA/HF RWKV backend가 `inputs_embeds + recurrent cache + gradient` gate를 통과해야 한다. 이 부분은 model/commit/custom kernel에 종속되므로 plugin 경계로 유지했다. Toy backend 결과를 논문 결과로 사용하지 않는다.

## 테스트

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src
for f in scripts/*.sh; do bash -n "$f"; done
```
