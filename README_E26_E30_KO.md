# Overlay reference implementation

이 디렉터리의 파일은 Codex가 live CATENA worktree에 통합할 **semantic reference**다.

- `src/catena/lm/`: matched gate surface, reference recurrence, data/schema/statistics/artifact helpers
- `experiments/`: fail-closed thin entry points
- `configs/`: v8.1 prospective protocol templates
- `tests/`: 최소 acceptance tests
- `scripts/`: dry-run과 승인 전 print-only launchers
- `tools/`: packet validator, hash audit, protocol lock, report summary

`reference_python` backend는 scientific MAIN을 실행할 수 없다. Live integration은 optimized scan과 repository dependency resolver를 추가한 뒤 parity를 통과해야 한다.
