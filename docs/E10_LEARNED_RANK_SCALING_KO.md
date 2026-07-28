# E10 - Learned Control Rank Scaling

## 질문

E03에서는 rank-8 operator가 residual을 제거할 수 있다는 oracle upper bound만 확인했다. E10은 transaction-conditioned low-rank controller가 실제로 학습되어 best-rank reachable floor에 접근하는지 검증한다.

## 데이터

Descriptor `z`에서 target operator를 생성한다.

\[
A(z)=U\operatorname{Diag}(c(z))V^\top
\]

Intrinsic rank는 1, 2, 4, 8, 16이며 train/test descriptor는 분리한다.

## 모델

Learned rank `r ∈ {1,2,4,8,16,32}`. MLP가 descriptor에서 left/right factors를 생성한다.

## Loss와 평가

- Train: matrix-entry MSE
- Test: normalized Frobenius MSE
- Oracle: per-example truncated-SVD best rank-r error
- Best-rank reachable-floor recovery: 각 rank controller의 optimization 진단
- Exact-target recovery: minimum sufficient rank 판정
- 핵심: 두 recovery, excess-over-oracle, minimum sufficient rank, parameter count

## Gate

- error가 rank에 따라 대부분 단조 감소
- recovery 0.95를 넘는 최소 learned rank가 intrinsic rank 이상이며 그 2배를 넘지 않음
- 최소 qualifying rank가 각 seed에서 intrinsic rank 증가에 따라 감소하지 않음
- 8-seed low-rank 대 high-rank effect sign-flip 통과

### Prospective pre-evaluation identifiability repair

`prospective_pre_evaluation_gate_identifiability_repair`는 evaluable E10 main
report가 생성되기 전에 고정했다. 기존 upper-bound-only 조건은 모든 demand
family가 learned rank 1로 풀리는 경우에도 통과할 수 있어 rank-scaling claim을
식별하지 못했다. 따라서 기존 recovery threshold, rank grid와
`max_rank_factor=2.0`을 변경하지 않고 다음 조건을 추가했다.

1. minimum qualifying rank의 lower bound는 intrinsic rank다.
2. 각 seed 안에서 minimum qualifying rank는 intrinsic rank가 증가할 때
   nondecreasing이어야 한다.
3. rank별 best-rank reachable-floor recovery는 optimization 진단으로만
   사용하고, 기존 0.95 threshold를 적용하는 minimum-rank qualification은
   intrinsic-rank exact oracle에 대한 exact-target recovery로 정의한다.

이 분리가 필요한 이유는 rank-1 controller가 자신의 best-rank-1 floor에
도달하면 reachable-floor recovery가 1이 될 수 있지만, rank가 큰 target의
나머지 성분을 복구하지 못했으므로 minimum sufficient rank가 될 수는 없기
때문이다.

이 변경 전 생성된 다음 두 run은 `run_manifest.json`만 있고 `report.json`이
없는 incomplete run이다. 원본 directory는 보존하되 scientific evidence로
사용하지 않는다.

| Run ID | 상태 | Evidence |
|---|---|---|
| `20260727T180703.792069Z` | `NO_REPORT` | ineligible |
| `20260727T180747.984294Z` | `NO_REPORT` | ineligible |

### Artifact-contract amendment

최초 protocol lock 뒤 시작한 `20260727T183537.502519Z`도 평가 결과가
생성되기 전에 중단되었다. 이 directory에는 schema-v2 provenance
(`run_manifest.json`, `config.resolved.yaml`, `environment.json`)만 있고
`report.json`, metric row, checkpoint가 없다. 따라서 역시 claim에 사용할
수 없는 incomplete run으로 보존한다.

본 실행 전에 model checkpoint와 checkpoint SHA-256을 각
`seed × intrinsic-rank × learned-rank` metric row에 기록하도록 source만
보강했다. Config, rank grid, metric, threshold와 claim gate는 변경하지
않았다. 이 순수 artifact-contract 보강은 별도 V2 lock으로 고정하며,
기존 V1 lock은 수정하지 않는다.

| Run ID | 상태 | Evidence |
|---|---|---|
| `20260727T183537.502519Z` | `NO_REPORT` | ineligible |

## 해석

성공하면 `richer control can help`라는 oracle 결과를 `learned control rank tracks demand rank`라는 학습 결과로 올릴 수 있다. Pretrained LM 또는 general algebra claim은 열지 않는다.
