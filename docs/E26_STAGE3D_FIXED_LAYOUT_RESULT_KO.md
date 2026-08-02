# E26 Stage-3D Fixed-Layout BF16 결과 요약

## 판정

```text
execution_status: COMPLETED_NUMERICAL_EVALUATION
disposition: STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY
evidence_tier: NON_EVIDENCE_NUMERICAL_PREFLIGHT
resource_preflight_started: false
scientific_e26a_started: false
```

Stage-3C의 arbitrary-layout 실패는 `KNOWN_BF16_AND_OPTIMIZER_LAYOUT_SENSITIVITY`로
그대로 보존했다. Stage-3D는 사전 고정한 단일 physical layout에서도 BF16
optimized/reference 및 BF16/FP32 reference 차이가 등록 relative-L2 한계
`0.007`을 넘는지 별도로 평가했다.

| 비교 | logits 평균 | state 평균 | gradient 평균 | 판정 |
| --- | ---: | ---: | ---: | --- |
| compiled BF16 vs reference BF16 | 0.02334 | 0.02807 | 0.03468 | 0/12 PASS |
| reference BF16 vs reference FP32 | 0.03389 | 0.04106 | 0.05019 | 0/12 PASS |

G0의 frozen E00--E25 artifact 2,062개, G1의 tied/dual physical-layout identity,
G2의 FP32 reference 12/12 report 및 132/132 nontrivial row는 통과했다. G3의
12개 case에서도 finite gradient, state metadata, clone/no-alias, cross-variant
identity와 compiled backend integrity는 모두 통과했다. 실패 원인은 오직 등록
BF16 numerical tolerance였다. G3가 0/12였으므로 G4 same-layout replay, G5
optimizer-step integrity 및 G4 의존 backend gate는 실행하지 않았다.

## Artifact

```text
run:
  /data/minjun_dev/CATENA/artifacts/
  e26_stage3d_fixed_layout_bf16_admissibility/20260802T145935.055835Z
source_commit: 47cbc68636367e32832c66ea57d1a827282ef447
protocol_sha256: ee18daba4291610afede1bb1e7a4c0b92570ecf0c0ba9d7e701593d9cf313480
report_sha256: 4c4528bf35052423896b29dbc12944e9ad5df3ec2f87410a9688417297a42650
status_sha256: d51ea2ce8c4648e103a392ce8a32ea731039af20eee433c55afdce7c0d0c2d21
artifact_audit_sha256: dac53f6cb220defbd13ef4416cb8f0798f5f0d6f88bc1624909828d8454f1d37
```

Artifact audit는 41개 등록 파일의 path/bytes/SHA, 12개 unique G3 row, 6개
dependency placeholder, canonical receipt, finite value, report/status/latest의
동일 disposition을 모두 확인했다.

## 개발 특기사항과 claim 경계

최초 admission 실패는 historical receipt의 생성 당시 HEAD를 동적 관측값으로
분리해 수리했다. 첫 GPU run `20260802T144040.692630Z`는 rich G3 row schema와
terminal disposition 전파 결함 때문에 authoritative `NOT_EVALUABLE`로 보존했다.
Threshold, seed, data와 layout은 바꾸지 않았고, 위 fresh run만 canonical 결과다.

허용되는 결론은 **현재 BF16 recipe가 고정 physical layout에서도 Stage-3D
admissibility를 충족하지 못했다**는 것이다. E26 과학 가설, dual/tied 성능,
batching-layout invariance, official GDN2/KDA 또는 언어모델 우월성은 평가되지
않았다. 다음 단계는 자동 실행이 아니라 별도 prospective
`FP32_HEAVY_RESOURCE_CONTRACT`의 연구·자원 설계 여부를 결정하는 것이다.
