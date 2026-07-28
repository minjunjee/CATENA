# E11 - Representation-Control Co-adaptation

## 질문

End-to-end model은 representation basis를 회전시켜 diagonal control의 한계를 없앨 수 있는가?

## Demand family

1. Axis commuting
2. Common-rotated commuting
3. Transaction-dependent noncommuting

모두 같은 descriptor interface와 held-out transaction split을 사용한다.

## Controller

- fixed-basis diagonal
- learned shared-basis diagonal
- transaction-conditioned low-rank

## 예측

- Axis: fixed와 learned basis가 동등
- Common rotation: learned basis가 fixed penalty를 복구
- Noncommuting: learned shared basis에 residual이 남음
- Low-rank: noncommuting residual 감소

## Primary gate

세 interaction이 8 seeds에서 같은 방향이고 registered SESOI를 넘으며, axis family에서 collateral degradation이 없어야 한다.

## 해석

성공하면 fixed synthetic basis에 의한 artifact라는 반론을 줄인다. Representation learning이 common rotation은 흡수하지만 noncommutative family의 구조적 요구까지 제거하지 못한다는 결과다.
