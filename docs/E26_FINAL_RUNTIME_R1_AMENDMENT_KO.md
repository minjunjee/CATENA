# E26 Final official-runtime R1 운영 amendment

상태: `PROSPECTIVE_BEFORE_R1`

최초 canonical official-runtime audit의 두 variant는 GPU kernel 실행 전에
`only_gate_source_modified` source-cleanliness gate에서 차단됐다.

- Dual blocked receipt:
  `/data/minjun_dev/CATENA/artifacts/e26_final_official_runtime_dual_7cMMKrfF/runtime_audit.json`
- Projected-Tied blocked receipt:
  `/data/minjun_dev/CATENA/artifacts/e26_final_official_runtime_tied_GSscUGVt/runtime_audit.json`

두 receipt의 error는 동일하며, derived official checkout에 gate-only source diff
외에 Python bytecode cache가 존재했다. 해당 cache의 mtime은 2026-08-03
16:00:10--16:00:13 UTC로, canonical receipt 생성 시각 16:02 UTC보다 앞선다.
이는 직전 implementation validation이 만든 비-source cache contamination이다.
Checkpoint/model forward, loss, throughput 또는 scientific outcome은 실행·관측되지
않았다.

R1에서는 다음만 수행한다.

1. 두 최초 BLOCKED receipt를 변경하거나 삭제하지 않는다.
2. derived checkout의 `__pycache__` regular files만 명시적으로 unlink한다.
3. gate-only `lit_gpt/gdn2.py` diff의 SHA는 변경하지 않는다.
4. `PYTHONDONTWRITEBYTECODE=1`로 fresh Dual/Tied R1 namespaces에서 같은 command,
   checkpoint, threshold와 metric을 한 번 재실행한다.
5. R1이 실패하면 추가 repair/retry 없이 terminal `BLOCKED_OFFICIAL_RUNTIME`으로
   중단한다.

이 amendment는 scientific protocol, gate policy, checkpoint, dependency pin,
precision, tolerance 또는 claim wording을 바꾸지 않는다.
