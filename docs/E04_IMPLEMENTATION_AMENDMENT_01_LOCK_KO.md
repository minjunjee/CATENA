# E04 implementation amendment 01 lock

- Frozen at: `2026-07-27T05:46:43.486738Z`
- Amendment SHA-256: `54272d3814c3fce5bcdca79d44566df0e818104a463ea3a693f9aa09466e984f`
- Triggering dry run: `20260727T054548.378221Z`
- Triggering manifest SHA-256: `f717f52aa9766c07564fd1ee6f01e8afaf572356df8f49be520c345a6078e092`
- Triggering manifest status: `RUNNING`
- Triggering dry eligibility: `main=false`, `full=false`
- Completed E04 main runs before amendment: `0`
- E04 main namespace inspected before amendment: `false`
- Metric/config/protocol threshold change: `false`
- Permitted change: checkpoint-seed report keys are serialized as JSON strings

원본 protocol, 원본 protocol lock, 실패 dry artifact는 수정하지 않는다. 이
amendment와 lock을 runner가 검증하고, 변경 source를 새 E00으로 다시 고정한
뒤에만 E04를 재실행한다.
