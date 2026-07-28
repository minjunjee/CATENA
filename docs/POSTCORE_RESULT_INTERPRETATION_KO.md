# Post-core 결과별 해석

## E10

- rank threshold가 intrinsic rank와 함께 증가하고 oracle-normalized recovery가 높음: demand rank가 필요한 learned control rank를 예측한다.
- error는 감소하지만 minimal rank가 추적되지 않음: richer rank의 capacity effect만 주장한다.
- 단조성 실패: rank claim을 닫고 optimization/parameter matching을 점검한다.

## E11

- common rotation에서 learned basis 회복, noncommuting에서 residual, low-rank 회복: representation co-adaptation 이후에도 algebraic obstruction이 남는다.
- common rotation조차 회복하지 못함: training/orthogonal parameterization failure 가능성이 먼저다.
- noncommuting도 diagonal이 해결: family construction 또는 data leakage를 감사한다.

## E12

- 추가 freedom이 해당 demand family에서만 선택적 이득: architecture-demand lattice claim.
- 모든 family에서 uniformly 우수: parameter/optimization confound 가능성이 크며 causal lattice claim 금지.
- simpler demand degradation: freedom 추가의 trade-off를 보고하고 universal improvement claim 금지.

## E13

- E13a NO-GO: E13b를 실행하지 않는다. floor, gap, retention, throughput 중 실패 원인을 분리한다.
- E13c supported: structured event sequence에서 dual control의 repeated-update 우위.
- E13c null: controlled one-step geometry가 sequence persistence로 자동 전이되지 않는 경계 결과.

## E14

- 5 seed×12 cell에서 affected-field correction gain과 untouched integrity
  guardrail 동시 통과: tested structured synthetic table-state proxy에서
  dual assimilation이 stale affected field를 교정한다.
- correction만 통과: destructive update 가능성 때문에 claim 금지.
- Whole-table gain은 normalization상 descriptive metric이며 primary gate가
  아니다.
- Untouched retention은 oracle localization 구조의 integrity assertion이지
  learned addressing 증거가 아니다.
- external-read/cached-snapshot 비교는 아직 구현되지 않았으므로 production break-even을 주장하지 않는다.

## E15

Official pinned backend의 PASS row만 architecture-transfer evidence다. Reference, dry-run, community checkpoint 결과를 official GDN2/KDA/KVEraser evidence로 부르지 않는다.
