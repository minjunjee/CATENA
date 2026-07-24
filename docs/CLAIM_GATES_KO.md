# 결과별 허용 주장

실험 결과보다 강한 문장을 쓰지 않기 위한 사전 claim ceiling이다.

| 관측 결과 | 허용되는 가장 강한 주장 |
|---|---|
| Exact teacher가 gold를 잘 풀지 못함 | backbone/task ceiling 분석. State transport 효과는 결론 내리지 않음 |
| Stale와 exact 차이가 없음 | 이 데이터 조건에서 coherence failure가 유도되지 않았다는 부정 결과 |
| H1만 성립 | 외부 state update 후 persistent recurrent state에 stale behavior가 남는다는 진단 |
| H1+H2 성립 | typed transaction/closure가 state update signal로 유효함 |
| Tx-only ≈ CATENA | 기존 state를 이용한 transport라는 주장은 불가; transaction-conditioned QA로 해석 |
| Shuffled closure ≈ correct closure | dependency closure의 semantic 기여 주장 불가 |
| Reset/retrieval ≈ CATENA | learned transport의 실용적 필요성/우위 주장 불가 |
| Generic soft-slot ≈ typed CATENA | typed structure의 추가 novelty 주장 불가 |
| H1-H3 성립 | compact native state transport가 exact-refresh behavior를 저비용으로 근사 |
| H4가 matched distillation-only보다 장기 drift를 줄임 | compositional consistency가 long-chain OOD generalization을 개선 |
| Transformer repair가 전 범위 우세 | recurrent-specific advantage를 철회하고 architecture boundary 분석만 유지 |
| RWKV가 긴 history/high update에서만 우세 | 해당 workload regime에 한정한 fixed-state 이점 |

## 메인 문장의 단계

1. **Diagnostic paper:** stale execution-state failure를 최초/체계적으로 측정했다.
2. **Representation paper:** verified typed transaction과 closure가 plain correction보다 더 나은 update signal이다.
3. **Method paper:** transaction slots를 native recurrent forward로 적용하면 coherence-cost Pareto가 개선된다.
4. **Compositional method paper:** 짧은 chain에서 학습한 consistency가 긴 chain의 drift를 줄인다.

하위 단계가 지지되지 않으면 상위 단계의 문장을 사용하지 않는다.
