# E15a-R1 official parity failure audit

**대상 run:** `20260728T062259.987278Z`  
**최종 disposition:** frozen `FAIL` 유지 · probe patch 없음 · official replication 차단

## 결론

등록된 tied-reduction과 BF16 parity 실패는 input, state layout, dtype
wiring 오류가 아니다. Official GDN2는 `b=β·1`, `w=β·1`에서 KDA를
복원한다고 명시하고, plugin도 동일한 broadcast를 적용한다. 두 official
chunk kernel의 solve precision 분기가 서로 달라 등록 tolerance 안의
수치 parity가 성립하지 않았다.

| 등록 check | 관측값 | 기준 | 판정 |
|---|---:|---:|---|
| Tied GDN2→KDA FP32 relative L2 | `1.378316e-4` | `≤1e-5` | FAIL |
| BF16 relative L2 | `5.903338e-3` | `≤5e-3` | FAIL |
| 나머지 full/chunk, gradient, state, intervention | 4/4 PASS | 사전 기준 | PASS |

## 원인 국소화

Tied parity는 `T≤16`에서 bit-exact이고, inter-subchunk solve가 시작되는
`T=17`부터만 차이가 생겼다. `T=96` intermediate 비교에서 `Aqk`, 누적
`g`, `qg`, `kg`는 exact였고 첫 residual은 `Akk=7.2059e-5`였다. 이후
WY와 output/state로 전파돼 output `1.3783e-4`, final state
`1.0991e-4`가 됐다.

GPU 3에서는 shared memory가 `101376 B`여서
`check_shared_mem=false`, `IS_TF32_SUPPORTED=true`다. 이에 따라 pinned
GDN2 solve는 IEEE, KDA solve는 TF32를 선택한다. BF16 probe도 active
operator input은 BF16, API가 요구하는 initial state만 FP32로 동일하게
구성됐다. `g` FP32 유지와 official q/k normalization 진단도 각각 약
`5.92e-3`으로 기준을 넘었다.

## Source reference

- Official reduction 정의:
  `/home/minjun_dev/CATENA_official/gdn2_upstream/lit_gpt/gdn2.py:54`
- Probe tied broadcast:
  `/home/minjun_dev/CATENA_official/plugins/catena_official_plugins/gdn2_gate.py:475`
- GDN2 precision branch:
  `/home/minjun_dev/CATENA_official/gdn2_upstream/lit_gpt/gdn2_ops/chunk_gdn2.py:366`
- KDA precision branch:
  `/home/minjun_dev/CATENA_official/gdn2_upstream/lit_gpt/gdn2_ops/chunk_kda.py:2643`

기존 report, result summary, freeze, source 및 threshold는 변경하지 않는다.
허용되는 기록은 “pinned official kernels에서 4/6 gate 통과, 전체 official
claim은 닫힘”까지이며 controlled-reference 결과를 official evidence로
승격하지 않는다.
