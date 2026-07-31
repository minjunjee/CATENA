# CATENA v8.1 Packet Semantic Integration

## Additive integration

`APPLY_PACKET.sh --check-only`는 73개 신규 경로와 충돌 0개를 보고했다.
해당 파일은 별도 worktree에만 추가됐다. 기존 tracked scientific file의
overwrite 또는 수정은 없다.

## 그대로 유지한 계약

- E26–E30 canonical numbering과 model name
- 14 prospective YAML의 metric, threshold, seed, disposition
- Dual/Projected-Tied의 maximal two-output gate head
- Reference implementation의 `NON_EVIDENCE_VALIDATION` 한계
- E27–E30 fail-closed dependency
- per-run immutable artifact와 one-page summary 계약

## Repository-specific integration이 필요한 항목

| Packet component | 통합 조치 |
|---|---|
| `recurrent_mixer.py` | reference loop는 보존하고 compiled/chunked scientific backend를 별도 경로로 추가 |
| `model.py` | recurrent matrix와 local-attention K/V ring buffer를 함께 clone하는 hybrid runtime state 추가 |
| `artifacts.py` | 기존 core IO/source fingerprint/lock metadata와 연결하고 completion-time provenance 추가 |
| tokenizer/corpus | byte/synthetic path는 smoke 전용으로 유지; 외부 16K tokenizer와 frozen token memmap을 hash-validated contract로만 허용 |
| schemas | packet의 7개 schema를 `schemas/v8_1/`에 additive 배치 |
| launch/status scripts | ambient Python 금지, approved environment와 dependency report를 명시 |
| E26a | reference dry-run과 optimized 100-step non-evidence smoke를 분리; MAIN authorization은 계속 차단 |

## 현재 claim boundary

구현·dry-run·smoke 결과는 모두 `NON_EVIDENCE_VALIDATION`이다. E26a scientific
MAIN은 prospective lock, canonical tokenizer/corpus, optimized backend manifest,
clean source commit, explicit `--allow-main`을 모두 요구하며 이번 작업에서는
실행하지 않는다.

