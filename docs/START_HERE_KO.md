# CATENA 4-GPU 실행 시작점

이 문서는 서버에 SSH로 접속한 뒤 어떤 순서로 무엇을 실행해야 하는지만 정리한다. 과학적 세부사항은 `EXPERIMENT_RUNBOOK_KO.md`, 결과별 주장 한계는 `CLAIM_GATES_KO.md`를 따른다.

## 0. 기본 원칙

- 네 GPU는 하나의 DDP job보다 **네 개의 독립 실험 lane**으로 사용한다.
- 메인 RWKV weight는 `fla-hub/rwkv7-2.9B-g1`과 동일 source PTH(`rwkv7-g1-2.9b-20250519-ctx4096.pth`)로 고정한다.
- H1/H2가 통과하기 전 H3를 시작하지 않고, H3가 통과하기 전 H4를 시작하지 않는다.
- Test split은 validation에서 slot 수와 checkpoint를 고른 뒤 한 번만 연다.

## 1. 설치와 서버 audit

```bash
cd CATENA_REALM2026_baseline_repo_v0.6.0
bash scripts/00_bootstrap_and_audit.sh
source .venv/bin/activate
source scripts/setup_paths.sh
```

확인할 파일:

```text
artifacts/logs/system_audit_*.txt
artifacts/logs/pip-freeze*.txt
```

## 2. RWKV backend 설치와 hard gate

```bash
FLA_REF=main bash scripts/install_rwkv_fla.sh
bash scripts/01_runtime_gates.sh
```

첫 설치에서는 최신 Blackwell/CUDA 13 수정이 필요할 수 있어 `main`을 허용한다. gate를 통과한 즉시 `artifacts/logs/fla-commit.txt`의 commit으로 이후 실행을 고정한다.

Hard gate:

1. 0.4B와 2.9B load
2. full/chunked parity
3. token/embedding ranking parity
4. recurrent cache clone 무변형
5. `inputs_embeds + cache`에서 encoder까지 finite gradient
6. Qwen KV crop과 suffix re-prefill parity

## 3. 데이터 생성

```bash
bash scripts/02_generate_and_validate_data.sh
```

이 단계가 만드는 고정 데이터:

```text
data/processed/pilot
 data/processed/main
 data/processed/stress
 data/processed/chains_main/chains
```

## 4. H1/H2: 학습 없는 진단

```bash
bash scripts/run_in_tmux.sh catena-h1h2 "bash scripts/03_h1_h2_pilot_4gpu.sh"
```

- H1: stale state와 exact refresh의 paired gap
- H2: plain/typed/closure/Tx-only/reset/retrieval

H1 stale gap이 없거나 exact teacher가 약하면 H3로 바로 넘어가지 않는다.

## 5. RWKV exact teacher

```bash
bash scripts/run_in_tmux.sh catena-teacher "bash scripts/04_build_rwkv_teacher_4gpu.sh"
```

Teacher cache는 candidate distribution만 저장한다. Backend cache object는 E01 serialization gate가 통과한 경우에만 별도 최적화한다.

## 6. H3: slot sweep, main seeds, 평가

```bash
PILOT_ONLY=1 bash scripts/05_train_h3_4gpu.sh
bash scripts/05_train_h3_4gpu.sh
```

Validation `C_joint`로 K=4/8/16 중 하나를 고르고 `configs/experiments/e07_h3_main.yaml`의 `num_slots`를 그 값으로 맞춘다.

```bash
bash scripts/06_train_h3_ablations_4gpu.sh
bash scripts/06_eval_h3_4gpu.sh
```

네 lane:

- GPU0/1/2: typed CATENA seeds 11/22/33
- GPU3: no-closure, 이후 untyped/generic control

## 7. Transformer 경계 실험

```bash
bash scripts/07_transformer_boundary_4gpu.sh
```

순서:

1. stale/append/capsule/suffix/full re-prefill
2. Qwen exact teacher
3. parameter-matched learned soft patch 3 seeds
4. learned patch main/stress evaluation

## 8. H4: composition

```bash
H3_INIT_CHECKPOINT=artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt \
  bash scripts/08_h4_composition_4gpu.sh
bash scripts/08_eval_h4_4gpu.sh
```

Seed 11을 자동으로 최선이라 가정하지 않는다. Validation에서 고른 H3 checkpoint를 명시한다.

## 9. 비용, tool call, clean rerun

```bash
bash scripts/09_profile_4gpu.sh
bash scripts/10_naturalized_toolcalls_4gpu.sh
CATENA_H3_CHECKPOINT=artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt bash scripts/10b_toolcalls_catena.sh
bash scripts/11_clean_rerun_and_freeze.sh
bash scripts/12_bundle_results.sh
```

최종 논문 표는 raw prediction JSONL에서 다시 계산한다. Summary JSON 수치를 수작업으로 복사해 조합하지 않는다.
