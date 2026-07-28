# E10–E21 결과 요약 coverage 감사

검증 시점: `2026-07-28`

| 항목 | 수 |
|---|---:|
| `report.json` + `run_manifest.json` 완료 run | 69 |
| Run 내부 `RESULTS_SUMMARY_KO.md` | 67 |
| Immutable sidecar summary | 2 |
| 미요약 완료 run | 0 |

Run 내부 summary 67개는 모두 UTF-8, 최대 60 lines·8,000 bytes 계약과
report/manifest hash 검사를 통과했고 다음 immutable index에 기록됐다.

- [E10–E21 run별 summary index](../artifacts/POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md)
- Index SHA-256:
  `96539e96b107c2b96b644c0ff0223905f47de66901140f727c7787fba586426a`

기존 artifact 불변 원칙 때문에 run 내부에 사후 파일을 추가하지 않은 두
초기 split official dry-run은 report SHA-256을 고정한 sidecar로 보존했다.

| Experiment / run | Sidecar |
|---|---|
| E15a `20260728T054350.258137Z` | [GDN2/KDA dry-run 요약](E15A_OFFICIAL_GDN2_KDA_DRY_RUN_RESULT_KO.md) |
| E15b `20260728T054350.379243Z` | [KVEraser dry-run 요약](E15B_OFFICIAL_KVERASER_DRY_RUN_RESULT_KO.md) |

Incomplete operational run은 완료 run 분모에 포함하지 않았다. E21의
외부 `SIGTERM` partial directory는 삭제·수정하지 않고
[운영 기록](E21_OPERATIONAL_RUN_LOG_KO.md)에 별도로 남겼다.
