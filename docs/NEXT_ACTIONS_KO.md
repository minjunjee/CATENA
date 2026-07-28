# CATENA post-core: 지금 당장 실행할 작업

현재 controlled core의 판정은 다음과 같이 고정한다.

- H1 behavioral reachability: supported
- H2 원본 E02: inconclusive, 변경 금지
- H2 prospective E02b: supported
- H3 원본 E03: categorical supported / calibration failed, 변경 금지
- H3 prospective E03b: quantitative calibration supported
- H4 functional mediation: supported
- H5 E05a 및 E05a-R1: no-go, 이번 제출에서 종료

새 실험은 기존 결과를 재판정하지 않고, **controlled geometry를 learned operator와 sequence memory로 확장**한다.

## 0. 확장팩 설치 직후

```bash
cd /home/minjun_dev/CATENA
source /home/minjun_dev/miniconda3/bin/activate catena-v6
set -a; source .env; set +a

python -m pytest -q tests/test_postcore_*.py
bash scripts/run_postcore_dry.sh /home/minjun_dev/CATENA /tmp/catena_postcore_dry
```

## 1. 즉시 실행 가능: 서로 독립적인 4개 lane

다음 네 실험은 H5나 official backend를 필요로 하지 않는다. 동시에 실행한다.

```bash
bash scripts/launch_postcore_wave1.sh /home/minjun_dev/CATENA
```

| GPU | 실험 | 질문 | 다음 단계에 미치는 영향 |
|---:|---|---|---|
| 0 | E10 learned-rank scaling | learned rank가 best-rank reachable floor를 실제로 따라가는가 | control-rank claim 강화 |
| 1 | E11 representation-control co-adaptation | representation을 함께 학습해도 noncommuting demand의 한계가 남는가 | fixed-basis 비판 제거 |
| 2 | E12 control-algebra lattice | magnitude, granularity, address, state-conditioning 자유도가 각 demand에서 선택적으로 필요한가 | top-tier architecture map |
| 3 | E13a-R1 paired sequence floor/throughput | paired tied-dual 차이와 affected floor, 공정한 forward ETA가 성립하는가 | E13b GO/NO-GO |

로그는 다음에 남는다.

```text
/data/minjun_dev/CATENA/artifacts/_launcher_logs/
```

## 2. 지금 즉시 실행 가능: evidence freeze

기존 H1-H5 결과의 `report.json`과 SHA-256을 단일 registry로 동결한다.

```bash
python experiments/e16_core_evidence_freeze.py \
  --config configs/e16_core_evidence_freeze.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

실제 artifact ID가 config 후보와 다르면 `configs/e16_core_evidence_freeze.yaml`의 후보 목록만 추가한다. 기존 report를 수정하지 않는다.

## 3. Prospective E13a-R1이 GO일 때만: sequence main

원본 E13a pilot이 아니라 E13a-R1의
`claim_gate.go_for_e13b=true`를 확인한다.

```bash
cat /data/minjun_dev/CATENA/artifacts/e13a_r1_sequence_floor_throughput/latest.json
```

GO이면 1차 wave를 실행한다.

```bash
bash scripts/launch_sequence_wave.sh /home/minjun_dev/CATENA 1
```

완료 후 2차 wave:

```bash
bash scripts/launch_sequence_wave.sh /home/minjun_dev/CATENA 2
```

완료 후 3차 wave:

```bash
bash scripts/launch_sequence_wave.sh /home/minjun_dev/CATENA 3
```

그 다음 aggregate:

```bash
python experiments/e13c_transactional_sequence_aggregate.py \
  --config configs/e13c_transactional_sequence_aggregate.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

## 4. E13c가 supported일 때만: plan continuation

```bash
python experiments/e14_plan_continuation.py \
  --config configs/e14_plan_continuation.yaml \
  --device cuda:0 \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

E14는 latest repaired E13c-R1 `MAIN/PASS/SUPPORTED` provenance가 봉인한
dual E13b-R1 5-seed checkpoint 전체만 사용한다. Primary gate는
prospective하게 동결한
affected-entity correction gain이며 whole-table gain은 descriptive다.
이 결과는 structured synthetic table-state proxy에 한정되고, 일반 agent
planning이나 production break-even을 주장하지 않는다.

## 5. 별도 환경에서 병렬 준비: official backend

기존 `catena-v6` 환경을 오염시키지 않는다. GDN2/KDA와 KVEraser는 별도 환경 또는 container에 설치하고, full commit SHA를 고정한다.

```bash
python experiments/e15_official_backend_gate.py \
  --config configs/e15_official_backend_gate.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --dry-run
```

실제 실행은 `CATENA_GDN2_REPO`, `CATENA_GDN2_COMMIT`, `CATENA_KVERASER_REPO`, `CATENA_KVERASER_COMMIT`과 plugin module이 모두 준비된 뒤에만 한다. Reference fallback은 허용하지 않는다.

## 중단 규칙

- E10: rank 증가에 따른 error 감소가 seed 전반에서 단조적이지 않으면 rank claim을 열지 않는다.
- E11: common rotation을 learned basis가 회복하지 못하면 representation co-adaptation pipeline을 먼저 수정한다.
- E12: 추가 자유도가 더 단순한 demand를 망가뜨리면 control-lattice claim을 열지 않는다.
- E13a-R1: paired contract, dual affected floor, tied-dual gap, retention,
  forward-only throughput, E13b-scale projected run/wave ETA 중 하나라도
  실패하면 E13b를 시작하지 않는다.
- H5는 다시 수리하지 않는다. E05b를 생성하지 않는다.
