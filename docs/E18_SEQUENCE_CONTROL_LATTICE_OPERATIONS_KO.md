# E18 Sequence Control Lattice 운영 가이드

이 문서는 동결된
`E18_SEQUENCE_CONTROL_LATTICE_PROTOCOL_KO.md`의 과학적 protocol을 변경하지
않는 실행 절차다. Launcher와 status checker는 `latest.json`을 입력 선택에
사용하지 않고, E18a namespace의 모든 run directory를 직접 검증한다.

## 실행 전제

```bash
cd /home/minjun_dev/CATENA
source /home/minjun_dev/miniconda3/bin/activate catena-v6

set -a
source .env
set +a

test "$CATENA_ARTIFACT_ROOT" = \
  /data/minjun_dev/CATENA/artifacts
```

E18은 아래 25개 cell을 seed-major, controller-lattice 순서로 등록한다.
Canonical index `i`의 GPU는 `i mod 3`으로 고정된다. 일부 완료 cell을
resume할 때도 나머지 cell의 GPU assignment는 바뀌지 않는다.

```text
seeds:
  101, 211, 307, 401, 503

variants:
  tied_scalar
  dual_scalar
  diagonal_value
  separate_address
  state_aware

GPUs:
  0, 1, 2
```

각 GPU에는 worker 하나만 생기며, 해당 GPU에 배정된 cell을 순차 실행한다.
따라서 같은 GPU에 여러 training process를 동시에 올리지 않는다.

## 1. Read-only 상태 확인

```bash
python scripts/check_e18_status.py \
  --repo-root /home/minjun_dev/CATENA \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

JSON이 필요하면 `--json`을 추가한다.

상태 의미:

| 상태 | 의미 | 자동 동작 |
|---|---|---|
| `MISSING` | 등록된 유효 MAIN run이 없음 | launch 대상 |
| `COMPLETED` | provenance가 검증된 MAIN run이 정확히 하나 | 명시적 skip |
| `INCOMPLETE` | MAIN artifact가 미완료이거나 hash/schema가 invalid | 전체 launch 차단 |
| `DUPLICATE` | 같은 seed×variant의 유효 MAIN run이 둘 이상 | 전체 launch 차단 |

`INCOMPLETE`와 `DUPLICATE`는 자동 삭제·선택·재실행하지 않는다. 출력된
run directory를 조사하고 별도 판단해야 한다. `latest.json`이 존재하거나
가리키는 위치가 달라도 위 판정에는 사용되지 않는다.

## 2. Deterministic schedule dry-run

```bash
python scripts/launch_e18_sequence_lattice_wave.py \
  --repo-root /home/minjun_dev/CATENA \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --dry-run
```

Dry-run은 25개 canonical cell의 GPU와 `RUN` 또는 `SKIP_COMPLETED`를
출력한다. Artifact directory, launcher log, subprocess와 GPU job을 만들지
않는다. 불완전 MAIN, duplicate 또는 동일 cell의 live process가 있으면
nonzero exit로 차단한다.

## 3. E18a MAIN 실행

Dry-run과 상태 확인이 통과한 뒤에만 명시적으로 실행한다.

```bash
python scripts/launch_e18_sequence_lattice_wave.py \
  --repo-root /home/minjun_dev/CATENA \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --execute
```

Launcher는 completed cell을 재실행하지 않고 missing cell만 canonical GPU
queue에 넣는다. 전역 file lock은 마지막 GPU worker가 종료될 때까지
유지되므로 두 launcher가 같은 grid를 동시에 예약할 수 없다. 각 E18a
subprocess는 실험 entry point가 생성하는 새 UTC run directory를 사용한다.

Launcher log:

```text
/data/minjun_dev/CATENA/artifacts/_launcher_logs/
└── e18_sequence_lattice_<UTC timestamp>/
    ├── launch_plan.json
    ├── gpu0_queue.json
    ├── gpu0_worker.pid
    ├── gpu0_worker.log
    ├── 00_tied_scalar_seed101.log
    └── 00_tied_scalar_seed101.result.json
```

GPU 1과 2에도 같은 구조가 생긴다. `launch_plan.json`은
`latest_pointer_used=false`, `aggregate_autorun=false`를 기록한다.

모니터링 예시:

```bash
LOG_DIR=$(
  find /data/minjun_dev/CATENA/artifacts/_launcher_logs \
    -maxdepth 1 -type d \
    -name 'e18_sequence_lattice_*' \
    -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)

tail -f "$LOG_DIR/gpu0_worker.log"
```

개별 training output은 같은 directory의 cell별 `.log`에 기록된다.

## 4. 25/25 provenance 확인

Worker가 모두 종료된 뒤 아래 명령이 성공해야 한다.

```bash
python scripts/check_e18_status.py \
  --repo-root /home/minjun_dev/CATENA \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --require-aggregate-ready
```

이 검사는 다음을 요구한다.

- 25개 seed×variant cell이 정확히 한 번씩 존재
- run당 등록된 48-row grid
- report, manifest, metrics, checkpoint와 protocol-lock hash 일치
- MAIN mode, `PENDING_AGGREGATE`, distractor path contract
- 총 25 source run과 1,200 source metric row
- incomplete, duplicate, live MAIN process가 없음

## 5. E18b aggregate는 별도 CPU 명령으로만 실행

Launcher는 E18b를 자동 실행하지 않는다. 위 25/25 검사가 통과한 뒤 별도
명령으로만 실행한다.

```bash
python experiments/e18b_sequence_control_lattice_aggregate.py \
  --config configs/e18b_sequence_control_lattice_aggregate.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

E18b 자체도 동일한 25-run provenance와 full paired grid를 다시 검증한다.
`latest.json`으로 source run을 선택하지 않는다.

## 장애 처리 원칙

- 실패한 MAIN artifact를 launcher가 지우거나 덮어쓰지 않는다.
- 완료 cell을 `--execute`가 자동 재실행하지 않는다.
- duplicate 중 하나를 시간순 또는 `latest.json`으로 임의 선택하지 않는다.
- 불완전 run이 생기면 상태 checker가 출력한 exact directory와 해당
  launcher log를 먼저 확인한다.
- 과학적 config, SESOI, margin, seed, grid 또는 동결 lock은 운영 문제를
  해결하기 위해 수정하지 않는다.
