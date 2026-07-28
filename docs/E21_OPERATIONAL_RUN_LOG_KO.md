# E21 운영 실행 기록

이 문서는 E21 protocol·metric·claim을 변경하지 않는 운영 이력이다.

## Seed 223 첫 시도

- Run: `e21a_structured_sequence_localization_transfer/20260728T073826.802651Z`
- 종료: process exit code `143` (`SIGTERM`)
- 생성 완료: config, environment, run manifest, `seed223_base.pt`
- 미생성: `report.json`, metrics JSONL, result summary
- 판정 자격: 없음 — incomplete operational run

불완전 run은 삭제·수정하지 않았다. 원인 메시지가 남지 않은 외부
`SIGTERM`이므로 scientific/model failure로 해석하지 않는다.

## Seed 223 두 번째 시도

- Run: `e21a_structured_sequence_localization_transfer/20260728T080054.097303Z`
- Seed: `223`
- Device mapping: physical GPU 1 → visible `cuda:0`
- Config·lock·seed·metric·threshold: 변경 없음
- 종료: process exit code `143` (`SIGTERM`)
- 생성 완료: config, environment, run manifest
- 미생성: checkpoint, `report.json`, metrics JSONL, result summary
- 판정 자격: 없음 — incomplete operational run

## Seed 331 첫 시도

- Run: `e21a_structured_sequence_localization_transfer/20260728T073834.028555Z`
- Seed: `331`
- Device mapping: physical GPU 2 → visible `cuda:0`
- Config·lock·seed·metric·threshold: 변경 없음
- 종료: process exit code `143` (`SIGTERM`)
- 생성 완료: config, environment, run manifest, `seed331_base.pt`,
  `seed331_separate_address.pt`
- 미생성: 나머지 checkpoint, `report.json`, metrics JSONL, result summary
- 판정 자격: 없음 — incomplete operational run

## Seed 557 첫 시도

- Run: `e21a_structured_sequence_localization_transfer/20260728T080152.348605Z`
- Seed: `557`
- Device mapping: physical GPU 3 → visible `cuda:0`
- Config·lock·seed·metric·threshold: 변경 없음
- 종료: process exit code `143` (`SIGTERM`)
- 생성 완료: config, environment, run manifest
- 미생성: checkpoint, `report.json`, metrics JSONL, result summary
- 판정 자격: 없음 — incomplete operational run

## 세 seed의 동일-protocol 재실행

Unified execution session의 외부 `SIGTERM`이 반복되어, terminal lifecycle과
분리된 detached `tmux` session에서 같은 명령을 처음부터 다시 실행했다.
E21 source runner에는 optimizer와 step을 포함한 resume contract가 없으므로
partial checkpoint에서 이어서 실행하지 않았다.

당시 journal/dmesg에는 OOM, killed-process, NVRM/Xid 또는 segfault가
없었고 RAM·VRAM·`/data` 여유도 충분했다. 세 process가 서로 다른
variant 단계에서 거의 동시에 같은 신호로 종료된 반면 GPU 0 process는
계속 실행되었다. Signal sender를 직접 특정할 audit record는 없으므로
scientific/model failure가 아니라 execution-session lifecycle failure라는
진단은 가장 강한 운영상 추론으로 기록한다.

| Seed | 새 run | Device |
|---:|---|---|
| 223 | `20260728T080724.810142Z` | `cuda:1` / physical GPU 1 |
| 331 | `20260728T080724.888411Z` | `cuda:2` / physical GPU 2 |
| 557 | `20260728T080724.909238Z` | `cuda:3` / physical GPU 3 |

세 재실행 모두 config·lock·seed·metric·threshold를 변경하지 않았다.

최종 E21 aggregate와 freeze에는 완료된 explicit source run만 전달하며,
incomplete run을 자동 검색하거나 결과 행으로 혼합하지 않는다.
