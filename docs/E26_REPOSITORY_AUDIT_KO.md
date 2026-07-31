# E26 Repository and Frozen-Evidence Audit

감사 시각은 2026-07-31 UTC이며, E26–E30 구현 전 상태를 기록한다.

## 결론

Phase A의 중단 조건은 발견되지 않았다. 신규 코드는
`/home/minjun_dev/CATENA_E26`의 별도 worktree에서만 개발하며,
live `/home/minjun_dev/CATENA`와 기존 scientific artifact는 수정하지 않는다.

| 항목 | 관찰값 | 판정 |
|---|---|---|
| Live repository | `/home/minjun_dev/CATENA` | PASS |
| Live branch / HEAD | `main` / `adfdeaf9e87a8602a8e334915d87acb9ff25af39` | PASS |
| Dirty state | clean | PASS |
| Post-E21 lineage | `c23986c12a199024a30fecdf94ae1bb55f67c071`이 HEAD ancestor | PASS |
| E26 worktree | `/home/minjun_dev/CATENA_E26` | PASS |
| E26 branch | `exp/e26-autoregressive-lm` | PASS |
| Artifact link | `artifacts -> /data/minjun_dev/CATENA/artifacts` | PASS |
| Active E22–E30 MAIN process | 없음 | PASS |
| Existing E26–E30 tracked path | 없음 | PASS |
| Overlay exact-path collision | 0 / 73 | PASS |

## Frozen evidence

기존 verifier
`scripts/verify_pre_e22_artifacts.py`와
`docs/POST_E21_PREIMPLEMENTATION_ARTIFACT_SHA256.json`을 사용해
canonical artifact root를 다시 읽었다.

| 항목 | 기대 | 관찰 | 판정 |
|---|---:|---:|---|
| E00–E21 files | 1,329 | 1,329 | PASS |
| Aggregate SHA-256 | `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` | 동일 | PASS |
| Missing / unexpected / changed | 0 / 0 / 0 | 0 / 0 / 0 | PASS |

Preflight receipt는 live tree 밖의
`/tmp/catena_e26_preflight_frozen_hash.json`에만 기록했다.
E22–E25의 기존 disposition과 claim boundary는 변경하지 않는다.

## Packet

| 항목 | 관찰 | 판정 |
|---|---|---|
| ZIP SHA-256 | `1a5289217d1dbd0b761fce39b01fb06ce9d6012fc64569c8bf4e16e46fff81f7` | PASS |
| ZIP regular files | 149 | PASS |
| Manifest entries | 147 / 147 hash-valid | PASS |
| Prospective protocols | 14 | PASS |
| JSON schemas | 7 | PASS |
| Entry points | 14 | PASS |
| Packet unit tests | 30 / 30 in isolated import layout | PASS |

현재 `catena-v6`에는 `jsonschema`가 없고 live editable `catena` package가
standalone packet namespace를 선점하므로, unpacked packet validator를 그대로
실행하면 import/dependency 오류가 난다. 이는 packet content failure가 아니다.
통합 worktree에서는 기존 `src/catena/__init__.py` 아래에 `catena.lm`이
정상 배치되며, validation dependency는 `/tmp` 격리 경로에서 사용한다.

## 기존 convention 재사용

- Run lifecycle: `experiments.common.initialize_run/finalize_run`
- IO and SHA: `catena.core.io`
- Source/config provenance: `catena.core.provenance_v61`
- Prospective lock and artifact metadata:
  `catena.post_e21.contracts`
- Seed-level inference: `catena.eval.statistics`,
  `catena.eval.postcore_metrics`
- Device resolution: `catena.systems.device`

신규 `catena.lm` helper는 위 계약을 확장하되 기존 module을 수정하지 않는다.

