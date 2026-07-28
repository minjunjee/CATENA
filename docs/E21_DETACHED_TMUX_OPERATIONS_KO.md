# E21a detached tmux 운영 도구

`scripts/manage_e21_source_tmux.py`는 장시간 E21a source run을 unified
terminal lifecycle과 분리하기 위한 **운영 전용** 도구다. E21 protocol,
config, metric, threshold, claim wording 및 기존 artifact는 수정하지 않는다.

## 고정 범위

| 항목 | 허용값 |
|---|---|
| Source entry point | `experiments.e21_structured_sequence_localization_transfer` |
| MAIN seed | `113, 223, 331, 449, 557` |
| Physical GPU | `0, 1, 2, 3` |
| Python | `catena-v6` |
| Artifact root | `/data/minjun_dev/CATENA/artifacts` |

도구는 source entry point를 `--seed`, `--device cuda:0`과 함께 호출하고,
선택한 physical GPU 하나만 `CUDA_VISIBLE_DEVICES`로 노출한다. Artifact
run directory는 도구가 만들지 않는다. 기존 entry point의
`initialize_run`만 새 UTC directory를 만든다.

## 상태 확인

```bash
cd /home/minjun_dev/CATENA

/home/minjun_dev/miniconda3/envs/catena-v6/bin/python \
  scripts/manage_e21_source_tmux.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  status
```

기계 판독용 출력:

```bash
/home/minjun_dev/miniconda3/envs/catena-v6/bin/python \
  scripts/manage_e21_source_tmux.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  status --json
```

상태 판정은 `latest.json`을 사용하지 않는다. Completed run은
`report.json`, `run_manifest.json`, metrics, 1-page Korean summary와 hash
contract를 확인한다. Running source process와 이 도구가 만든 named tmux
reservation을 함께 확인한다.

## 실행 전 계획만 확인

`--execute`가 없으면 파일과 process를 만들지 않는다.

```bash
/home/minjun_dev/miniconda3/envs/catena-v6/bin/python \
  scripts/manage_e21_source_tmux.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  launch --seed 223 --gpu 1
```

다음 경우 launch를 거부한다.

- 등록되지 않은 seed 또는 GPU
- 같은 seed의 active source process 또는 tmux reservation
- 같은 seed의 completed MAIN
- 선택 GPU에서 다른 E21a source가 active
- 같은 seed의 incomplete run이 있는데 retry를 명시하지 않은 경우
- locked config/protocol hash 또는 `catena-v6` Python 불일치

## SIGTERM 이후 fresh retry

Partial checkpoint resume contract가 없으므로 incomplete run에서 이어서
학습하지 않는다. Incomplete directory는 삭제·수정하지 않고 그대로
보존한다. 새 UTC run을 의도적으로 시작할 때만 다음 flag를 사용한다.

```bash
/home/minjun_dev/miniconda3/envs/catena-v6/bin/python \
  scripts/manage_e21_source_tmux.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  launch --seed 223 --gpu 1 \
  --allow-incomplete-retry \
  --execute
```

실행 시 session 이름은
`catena_e21_seed<seed>_<UTC timestamp>` 형식이다. 운영 기록은 새
`_launcher_logs/e21_source_tmux_<UTC timestamp>_seed<seed>/` 아래
`launch_record.json`, `worker.log`, `result.json`으로 생성된다. 이 파일은
새로만 만들며 덮어쓰지 않는다.

모니터링 예:

```bash
tmux list-sessions
tmux attach -t catena_e21_seed223_<UTC_timestamp>
```

Session attach는 선택 사항이다. Worker가 끝나면 해당 tmux session은
종료되며 `result.json`에 return code와 새 source run directory가
기록된다. Aggregate는 자동 실행하지 않는다.

## 현재 실행에 대한 주의

등록 seed가 이미 active 또는 completed라면 이 도구를 다시 실행하지
않는다. 먼저 `status`로 확인한다. 기존 manual tmux session도 실제 E21a
process command를 통해 active로 탐지되므로 중복 seed launch가 차단된다.
