# CATENA 4-GPU 실험 매트릭스

이 표는 **무엇을 먼저 돌리고, 무엇을 학습하며, 어떤 수치로 다음 단계로 넘어갈지**를 한 페이지에서 확인하기 위한 실행 기준이다. 세부 정의와 claim ceiling은 `EXPERIMENT_RUNBOOK_KO.md`와 `CLAIM_GATES_KO.md`를 따른다.

| 순서 | 연구 질문 | 모델과 데이터 | 학습 또는 비교 | Loss | 핵심 평가 | 다음 단계 gate |
|---|---|---|---|---|---|---|
| E00 | 서버 환경이 고정됐는가 | 모델 없음 | driver/CUDA/PyTorch/GPU/topology/storage audit | - | 4 GPU, BF16, cu130, compiler, manifest | audit 통과 |
| E01 | state/cache adapter가 올바른가 | RWKV 0.4B/2.9B, PTH cross-check, Qwen 3B; smoke episode | full-vs-chunk, token-vs-embedding, clone, gradient, KV crop | - | ranking parity, max abs error, finite gradient | main RWKV와 Qwen hard gate |
| E02 | 데이터가 문제와 누출을 분리하는가 | UpdateBench pilot/main/stress/chain | schema/transaction/query/closure validator | - | gold consistency, affected/retention 존재, split hash | split freeze |
| E03 / H1 | stale state가 실제 오류를 만드는가 | RWKV 2.9B; pilot→main test | stale reuse vs exact refresh | - | gold, own-oracle, teacher-correct, C_update/C_retain/C_joint, stale rate, KL | exact ceiling + paired stale gap |
| E04 / H2 | 어떤 update signal이 필요한가 | RWKV 2.9B; 동일 episode | plain, typed, closure, shuffled, Tx-only, reset, retrieval, exact | - | H1 metrics + token/update latency + Pareto | closure 효과와 비누출 확인 |
| E05 | exact teacher를 재사용할 수 있는가 | RWKV 2.9B; main train/val | 4-way sharded exact refresh scoring | - | finite candidate logits, teacher accuracy, hashes | teacher cache complete |
| E06 / H3a | transport가 학습 가능한가, K는 얼마인가 | frozen RWKV 2.9B + 5-20M encoder; main train/val | typed K=4/8/16, generic K=8; seed 11 | 1.0 KL_aff + 0.6 KL_ret + 0.5 CE_gold | val C_joint, KL, finite grad, update latency | validation으로 K 고정 |
| E07 / H3b | CATENA가 Pareto를 개선하는가 | selected K; seeds 11/22/33; main/stress | typed main, no-closure, untyped, generic; text/reset/retrieval/exact baselines | E06와 동일 | 3-seed C_joint, KL, stale rate, p50/p95 update, state bytes, Pareto | H3 claim gate |
| E08 | Transformer repair와 비교한 경계는 무엇인가 | Qwen2.5-3B; same main/stress | stale, append, capsule, oracle suffix re-prefill, full re-prefill, learned soft patch 3 seeds | RWKV와 같은 H3 loss | own-oracle, C_joint, TTFA, decode, KV bytes, recomputed tokens | architecture regime map |
| E09 / H4 | composition이 long-chain drift를 줄이는가 | H3 init; chain train 1-2/test 4/8/16; 3+3 seeds | composition/path model vs matched distillation-only control | 1.0 KL_exact + 0.5 KL_ret + 0.5 KL_path + 0.25 CE; control은 KL_path=0 | chain별 C_joint/KL, drift slope, retention decay, stale recurrence | H4 claim gate |
| E10 | 실제 시스템 비용은 무엇인가 | RWKV/Qwen; 1K/4K/16K/32K | warm-up 5, measured 30 | - | prefill/update/TTFA/decode p50,p95; state/KV/peak bytes | final Pareto |
| E11 | candidate coherence가 tool action으로 이어지는가 | stress/naturalized subset | RWKV closure/exact/CATENA; Qwen closure/full | - | JSON validity, tool/arg exact, simulator success, stale field | action-level validation |
| E12 | 결과와 주장이 재현되는가 | validation-selected configs only | clean rerun, bootstrap, bundle | - | raw predictions→tables/figures, hashes, CI | anonymous submission package |

현재 E00은 dependency, native CUDA/PyTorch BF16, storage integrity와
repository validation을 모두 통과했다. E01 인프라 선행조건은 열렸으며,
다음 분기점은 pinned RWKV/Qwen backend의 모델별 runtime hard gate다.

## 4-GPU 배치 원칙

- **Teacher 또는 inference shard:** 같은 config를 데이터 4 shard로 분할한다.
- **학습:** GPU 0/1/2는 seed 11/22/33, GPU 3은 matched control 또는 structural ablation을 맡는다.
- **DDP는 기본값이 아니다.** 3B backbone과 작은 encoder는 한 GPU에 적재하고 네 독립 run을 동시에 수행하는 편이 연구 일정에 더 유리하다.
- **Test는 선택에 사용하지 않는다.** K, checkpoint, threshold는 validation에서 고정한 뒤 main/stress test를 연다.
