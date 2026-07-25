# Baseline status v0.6.0

## E00 현재 판정

- 전체 판정: **PASS** — E01 모델 runtime gate의 인프라 선행조건 충족
- 통과: 실제 Conda/Python/package identity, 8-GPU host inventory 중 GPU
  0–3 4-lane 할당, driver/CUDA compiler, `sm_120` kernel, 네 lane의
  네이티브 CUDA 및 PyTorch BF16 계산과 UUID/PCI identity, state/model
  cache 무결성, 33개 pytest, source/config/package/Conda manifest
- 경고: 저장소 가용 공간 약 25 GiB(권장 128 GiB 미만), host의 추가
  GPU 4–7, 작업 중인 dirty source snapshot
- 상세: `docs/E00_RESULT_SUMMARY.md`,
  `artifacts/profiles/e00_audit/latest.json`

## 정적·stdlib 테스트에서 검증된 부분

- Episode/chain data generation and validation
- Transaction rendering and policy logic
- Coherence metrics and shard merge
- Mock model end-to-end smoke
- Transformer suffix-reprefill cache-crop unit test
- Transaction slot encoder and loss unit tests

## 대상 서버에서 확인해야 하는 부분

- `fla-hub/rwkv7-2.9B-g1`과 동일 source PTH(`20250519`)의 logits/ranking 교차 검증

- FLA/HF RWKV-7 model loading on CUDA 13/Blackwell
- Recurrent cache clone/crop semantics for the pinned model revision
- Differentiable `inputs_embeds` forward through the RWKV recurrent cache
- 2.9B runtime throughput and memory

## 워크샵 이후 확장 범위

- Learned transaction writer
- Graph database or retrieval training
- Full backbone fine-tuning
- RL, branch/merge, online deployment
