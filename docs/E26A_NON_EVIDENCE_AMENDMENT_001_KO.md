# E26a Non-Evidence Numerical Amendment 001

## 범위

이 문서는 Scientific MAIN 이전의 단일-GPU non-evidence validation에서 발견한
구현 수준 수치 문제를 기록한다. 기존 run과 threshold는 수정하거나 재판정하지
않는다.

## 원본 run

- Run:
  `/tmp/catena_e26_dry_gpu_smoke_20260731T0701Z/e26a_operator_data_gate/20260731T070224.840049Z`
- Source commit: `28f6b868e9444ad52a6a09b97559691563561d02`
- Disposition: `NON_EVIDENCE_100_STEP_SMOKE_NUMERICAL_FAIL`
- Recurrent carry relative L2: `0.010864716954529285` (criterion `<= 0.007`)
- Attention carry relative L2: `0.007040057331323624` (criterion `<= 0.007`)

FP32 reference/optimized parity, BF16/FP32 operator parity, output carry,
gradient, intervention confinement, graph-break, fallback 검사는 통과했다.

## 원인 분리

동일 checkpoint의 추가 non-evidence 진단에서 FP32 full-vs-continuation
runtime-state 오차는 약 `1e-6` 이하였다. BF16에서만 split에 따라 오차가
누적됐으며, FFN output projection을 FP32로 계산하면 동일 비교가 정확히
일치했다. 이는 recurrence equation이나 state-carry indexing 문제가 아니라
sequence row 수에 따라 달라지는 BF16 GEMM reduction 경로의 누적 오차다.

## Prospectively fixed repair

1. FFN input projection과 activation은 기존 BF16 autocast를 유지한다.
2. FFN output projection만 FP32 accumulation으로 계산하고 residual stream
   dtype으로 복귀한다.
3. Model equation, parameter surface, initialization, data, seed, metric, threshold,
   warmup/measured steps는 변경하지 않는다.
4. Backend manifest에 graph-break/fallback count를 top-level에도 기록하여
   readiness validator와 artifact contract를 일치시킨다.
5. 수정 후 새 source commit과 새 run namespace에서 동일 smoke를 다시 실행한다.

이 amendment는 scientific evidence가 아니며 E26a, E26b 또는 LM claim을 열지
않는다.
