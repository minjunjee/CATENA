# E24 이론 스트레스 프로토콜

## 1. 범위와 현재 실행 상태

E24는 E00--E21 및 완료된 H1--H5 증거를 수정하거나 재해석하지 않는 별도
post-core 이론 스트레스다. E24a와 E24b는 각각 독립된 Python 진입점, YAML
설정, 프로토콜 lock, 실행 디렉터리를 가진다. 두 실험의 현재 정적 lock은
`protocol_frozen_before_main=true`, `main_execution_started=false`,
`main_authorized_by_default=false`,
`main_requires_explicit_allow_main=true`이다. 기본 호출은 중단되며 dry-run은
CPU와 `/tmp` 아래의 새 디렉터리에서만 실행된다. main은 사용자가
`--allow-main`과 `--dependency-root`를 함께 명시하고 모든 dependency 검증을
통과할 때만 실행된다. 이 구현ㆍ동결 단계에서는 main을 실행하지 않는다.
dry-run 결과는 과학 주장에 사용할 수 없다.

일반 E24 mock은 구현 참고자료일 뿐이며 이 프로토콜, 공식 backend 또는
언어모델 증거가 아니다. E24 구현은 그 파일을 실행ㆍ수정ㆍ이동하거나 결과로
인용하지 않는다.

## 2. E24a: approximate-rank 스트레스

차원 64의 제어된 행렬족에 다음 특이값 스펙트럼을 사전 등록한다.

- exponential: `tau = 1.5, 3, 6, 12`
- power law: `exponent = 0.75, 1, 1.5, 2`
- low-rank plus noise: base rank `2, 8, 16`과 tail noise
  `0, .01, .05, .10`의 곱집합
- 평가 rank: `1, 2, 4, 8, 16, 32`

각 스펙트럼은 Frobenius norm 1로 정규화한다. seed 안의 모든 spectrum
instance는 같은 deterministic left/right basis를 공유한다. 스펙트럼에서 family
label 없이 다음 16차원 descriptor를 만든다.

- 사전 등록 singular-value index 8개의 log 값
- 사전 등록 rank 6개의 cumulative energy
- dimension으로 정규화한 effective rank와 stable rank

각 seed마다 exponential, power-law, low-rank-plus-noise 중 한 family 전체를
holdout하는 세 fold를 결과와 무관하게 고정한다. 각 fold와 등록 controller
rank마다 E10/E10b의 `LowRankOperatorController` 패턴을 재사용하고 AdamW로
나머지 두 family의 descriptor--clean operator pair만 학습한다. descriptor
표준화 통계도 train family에서만 적합한다. holdout family label은 feature가
아니며 그 family의 target 또는 평가 outcome은 optimizer에 전달하지 않는다.
실제 model state, optimizer trace, seed, train/test instance ID를 fold/rank별
checkpoint에 보존한다.

holdout descriptor에서 생성한 rank-controller prediction tensor와 metadata를
artifact로 먼저 기록한 뒤에만 clean holdout target을 join한다. 주 결과는
이 learned OOD prediction의 normalized Frobenius error다. rank별 oracle floor는
Eckart--Young tail energy를 전체 에너지로 나눈 값이고,
`OOD learned excess = OOD learned normalized error - oracle floor`다. 수치
허용오차 밖에서 learned error가 oracle보다 낮으면 실행 오류다.

effective rank는 squared-singular-value energy 분포의 Shannon entropy를
지수화한 값이고, stable rank는 Frobenius energy를 최대 singular-value
energy로 나눈 값이다. 상대 잔차가 `epsilon=.05` 이하가 되는 최소 등록
rank를 oracle과 learned 양쪽에서 구한다. 등록 grid에서 해가 없으면
`unresolved`이며 match로 세지 않는다.

tail noise가 0인 low-rank-plus-noise 셀만 exact-rank reference다.
exponential, power law, 그리고 양의 tail noise 셀은
`construction_spectrum_stress`로 보고한다. 세 family 각각에서
epsilon-minimal rank match fraction과 mean excess gate를 계산하고 모든 family
fold가 통과해야 제한된 OOD spectrum-family transfer disposition을 지지한다.
별도의 noisy empirical SVD는 target별 직접 factorization 진단이며
`primary_estimand=false`로 기록하고 family-transfer 근거로 사용하지 않는다.
OOD 주장의 범위는 이 세 synthetic spectrum family와 seed별 shared basis에만
한정된다. 새 basis/geometry, 다른 spectrum family, 자연언어 작업으로의
zero-shot transfer를 주장하지 않는다.

## 3. E24b: behavioral-attainability 스트레스

다음 유한 설계를 결과와 무관하게 사전 등록한다.

- target/teacher relative noise factor support: `0, .01, .05, .10`
- noise condition 10개: clean, 세 target-only, 세 teacher-only, 세 matched-positive
- readout lambda: `.25, .5, .75`; horizon: `1, 4, 8`
- readout: linear, fixed block-local nonlinear residual MLP
- demand: axis-commuting, common-rotated-commuting, noncommuting
- controller: fixed diagonal, shared-basis diagonal, rank-8, full
- geometry: baseline, unseen key-correlation, unseen operator norm, unseen key load

nominal operator의 첫 75% output row만 affected block이고 나머지 retained row는
정확히 0이다. target noise는 affected row 안에서만 nominal operator를
perturb하고, 그 realization을 이후 평가할 **clean application target**으로
고정한다. teacher noise는 이 clean target에 독립적으로 더해지며 retained row도
오염할 수 있다. controller는 noisy teacher만 projection/fitting에 사용하지만
application outcome은 항상 clean target state와 비교한다. teacher factor만
바뀐 paired cell에서 clean-target hash는 같고 teacher hash는 달라야 한다.

`axis_commuting`은 affected block의 diagonal family다.
`common_rotated_commuting`은 하나의 affected-row orthogonal basis에서
동시 diagonalizable한 대칭 family이며 step별 diagonal shape는 비례하지 않는다.
`noncommuting`은 step마다 다른 basis를 쓴다. geometry는 단순 이름표가 아니라
입력 key covariance와 target norm에 직접 적용한다. baseline은
`rho=0, norm=1, high-load fraction=.5`; 세 OOD block은 각각
`rho=.75`, `norm=1.5`, high-load fraction `1.0` 하나만 바꾼다. high-load 밖의
key scale은 `.25`이고, temporal input block은 서로 직교하여 각 step covariance가
등록 key covariance와 정확히 일치한다. manifest/raw row에 profile 값, realized
correlation/load와 covariance hash를 기록한다. seed별 target/noise/input/readout
기초 난수는 geometry profile 사이에서 paired되어, key-correlation과 key-load
block은 operator realization을 바꾸지 않고 operator-norm block만 정확한 배율을
적용한다.

linear readout은 affected와 retained state slice 안에서 각각 QR whitening을
적용하며 slice 사이를 섞지 않는다. nonlinear readout도 각 slice 안에서만
작동하는 deterministic tied-orthogonal two-layer tanh residual MLP이고 학습하지
않는다. 각 slice의 clean-target--controller output MSE를 따로 계산하고 outcome을
정확히 `lambda * affected_mse + (1 - lambda) * retained_mse`로 정의한다.
lambda를 전체 차이에 먼저 곱하고 제곱하는 global scaling은 금지한다.
midpoint-Jacobian proxy와 Lipschitz upper bound도 두 component를 따로 계산한 뒤
같은 weight로 합친다.

predictor artifact에 들어가는 feature는 noisy teacher와 fitted controller 사이의
operator residual, linearized regret, Lipschitz bound 및 등록 nuisance factor뿐이다.
clean target hash, clean application error와 class bound는 feature row에서 금지한다.
각 fold에는 train clean outcome만 주고 test outcome은 주지 않는다. teacher-side
feature JSONL, test prediction JSONL과 predictor checkpoint를 먼저 기록한 뒤에야
clean test outcome을 join한다.

leave-one-demand-family, leave-one-controller-class와 네 개의 의미 있는
leave-one-geometry-block fold를 고정한다. held-out label은
`baseline_geometry`, `unseen_key_correlation`, `unseen_operator_norm`,
`unseen_key_load`다. log behavioral MSE 척도에서 `R2`, `RMSE`, `MAE`, Pearson,
tie-aware Spearman, calibration slope/intercept를 보고한다. held-out
level-by-seed mean actual/predicted pair도 family-level scatter artifact로 남긴다.
또한 lambda, noise condition, horizon 각 level에서 observed error, class-specific
lower bound/excess와 affected/retained MSE의 seed별 평균을 descriptive sensitivity
artifact로 남긴다. 이 marginal summary는 인과 효과로 해석하지 않는다.

prospective gate의 upper unit은 episode row가 아니라 seed다. main의 8개 seed별
metric을 먼저 계산하고 seed cluster만 2,000회 replacement bootstrap하여 95%
interval을 만든다. gate는 R2/Pearson lower bound와 normalized-RMSE upper bound를
사용한다. episode-row bootstrap은 금지한다. dry-run의 32회 단일-seed interval과
모든 gate 수치는 과학 평가가 아니다.

optimization gap은 universal full-controller zero floor를 쓰지 않는다. 각 cell의
clean target, key covariance, lambda와 같은 controller class만으로 row/key-weighted
analytic projection을 만들고, readout co-Lipschitz constant를 적용한
`controller_specific_clean_target_analytic_behavioral_lower_bound`를 outcome과
독립적으로 먼저 정한다. observed application outcome은 이 bound 계산에 절대
들어가지 않는다. 이 bound row는 observed application metric을 계산하기 전에
freeze하여 별도 JSONL에 기록한다. clean analytic controller의 실제 attainable
error는 별도 진단으로 기록한다. excess는 `observed application error -
class-specific lower bound`이며 bound와 excess는 predictor feature가 아니다.

main claim disposition은 전체 OOS gate와 별도로 두 subset gate를 계산한 뒤
다음 순서로 고정한다.

1. 전체 gate와 `teacher_noise>0 AND nonlinear AND H>1` subset가 모두 통과하면
   `BROAD_NOISY_NONLINEAR_MULTISTEP_PASS`.
2. 그렇지 않고 `linear AND H=1` subset가 통과하면 `ONLY_LINEAR_H1_PASS`.
3. 나머지는 `CONSTRUCTION_ROBUST_PREDICTION_FAILURE`.

## 4. lock 및 불변 artifact 전략

E24a와 E24b lock은 schema version 1, 실험 ID, 프로토콜 버전, prospective
상태 플래그, 설정과 구현 파일의 SHA-256 map을 포함한다. 실행 시작과 종료
직전에 모든 hash를 검증하고, 검증된 lock 원문을 실행 디렉터리의
`protocol_lock.json`으로 복사한다. 각 실행은 새 UTC 디렉터리만 만들며 기존
artifact를 덮어쓰지 않는다. `data_manifest.json`, raw/seed JSONL, checkpoint,
보고서에는 source/config/data/checkpoint/protocol hash를 남긴다. 각 run은
45줄 이하의 `RESULTS_SUMMARY_KO.md`도 생성하며, report에 path, SHA-256,
line count를 기록한다. dry-run 요약 상태는 반드시
`DRY_RUN_NON_EVIDENCE`이고 metric 또는 gate를 과학 결과로 해석하지 않는다.

`--dry-run`과 `--allow-main`은 상호 배타적이다. 둘 다 없거나, main에서
`--dependency-root`가 없거나, CPU가 아니면 artifact 디렉터리를 만들기 전에
중단한다. 실행 가능한 main 명령 형태는 다음과 같으며 사용자의 별도 명시적
지시 전에는 실행하지 않는다.

```bash
python experiments/e24a_approximate_rank_stress.py --allow-main \
  --dependency-root /data/minjun_dev/CATENA/artifacts --device cpu
python experiments/e24b_behavioral_attainability_stress.py --allow-main \
  --dependency-root /data/minjun_dev/CATENA/artifacts --device cpu
```

동결 뒤에는 metric, seed, gate, holdout, claim wording을 결과에 맞춰 바꾸지
않는다.

## 5. immutable evidence dependency

미래 E24 main은 다음 세 report를 모두 exact path, SHA-256, expected status로
검증해야 한다.

| Anchor | 상대 report path | SHA-256 | Expected status |
|---|---|---|---|
| H1 | `e01b_constrained_behavioral_reachability/20260726T152354.081239Z/report.json` | `8e1d16ca7763cec1e4e5b13d2b0f163f4015c8058ed7764871a6fbb5fa5ea6d6` | `PASS / SUPPORTED` |
| E10b | `e10b_floor_aware_rank_scaling/20260727T190906.272784Z/report.json` | `30f2f781bcc8528964602e0c66e1b61bb9d71a6ca5f964b833b2551c93b72484` | `PASS / SUPPORTED` |
| E11b | `e11b_scale_normalized_coadaptation/20260727T183004.928280Z/report.json` | `54015400029b3eae0367a1c12cb1dd717dee5a0568906f7d8e972c45bc4301b3` | `PASS / SUPPORTED` |

dry-run은 canonical artifact를 열지 않고 이 expectation만 manifest, report와
lock에 복제한다. explicit `--allow-main` 경로에서는 실행 전후로 세 report를
검증한다. missing, symlink, path escape, hash mismatch, JSON/status mismatch
중 하나라도 발견되면 실행 상태는 `BLOCKED_DEPENDENCY`이며 reference/mock
결과로 대체하지 않는다.

## 6. 주장 상한

승인된 main이 미래에 실행되더라도 E24a는 등록된 세 controlled spectrum
family와 seed별 shared basis 안의 descriptor-conditioned learned-controller OOD
transfer, E24b는 등록된 finite-dimensional construction의 OOS calibration만
지지할 수 있다. universal rank/reachability theorem, 새 geometry나 spectrum
family로의 transfer, parameter efficiency, causal identification, semantic 또는
자연언어 transaction, agent/planning transfer, 공식
recurrent-memory/KVEraser backend, pretrained language-model 주장은 모두 범위
밖이다.
