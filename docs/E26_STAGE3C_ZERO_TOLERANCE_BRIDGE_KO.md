# E26 Stage-3C zero-tolerance final-data bridge

Stage-3C는 기존 E26 V1의 source, split, tokenizer, transaction 및 resource
threshold를 변경하지 않는다. Zero-tolerance repair config는 data construction
전용이라 V1의 frozen-repository/resource 필드를 반복하지 않았으므로, GPU
preflight 전에 다음을 하나의 additive composite lock으로 결합한다.

```text
E26 data lock V1
  + zero-tolerance repair protocol/receipt
  + independently reconstructed readiness-v3
  = Stage-3C final-data lock
```

Composite lock은 repair protocol이 exact path와 SHA로 고정한 V1 payload를
복제하고 다음만 명시적으로 바꾼다.

- schema/status; V1 `repository.e26_worktree`는 보존하고 Stage-3C의 non-evidence
  execution worktree, commit, clean status와 executable-source aggregate는 별도
  `stage3c_execution` field에 기록
- repaired data root 및 exact receipt SHA
- final protected–train flag count `0`
- `execute_stage3c_non_evidence_preflight: true`

V1 `repository`, `resource_policy`, candidate order, numerical threshold,
tokenizer와 transaction contract는 byte-equivalent 구조로 유지한다. Human/AI
label, outcome metric 또는 main-test data는 사용하지 않는다. 이 lock의 claim
ceiling은 `PROTOCOL_IDENTIFIABILITY_ONLY`이며, numerical/resource preflight의
실행만 허용하고 feasibility 결과나 scientific E26a 실행 권한은 미리 열지 않는다.

Lock writer와 Stage-2 protocol builder는 bound input에서 payload 전체를 다시
구성해 exact equality를 요구한다. 따라서 lock hash를 다시 계산하더라도 V1의
threshold, stop policy, readiness/schedule binding 또는 zero-flag disposition을
변경한 payload는 통과하지 않는다.

Report-only Markdown commit은 executable-source inventory에서 제외되므로 lock
commit의 descendant이면서 source aggregate가 같은 경우에만 허용한다. Python,
YAML, JSON, shell 등 실행 입력이 달라지거나 Stage-2 `repo_root`와 execution
worktree가 다르면 fail-closed한다.

Stage-3 handoff가 `/home/minjun_dev/CATENA_E26`을 read-only로 보존하고
`/home/minjun_dev/CATENA_E26_STAGE3`를 실행 worktree로 지정했으므로,
non-evidence Stage-3C만 후자에서 실행한다. 원본 E26a config와 그 안의 canonical
scientific worktree 경로는 변경하지 않는다. 향후 scientific E26a는 별도 승인 후
`/home/minjun_dev/CATENA_E26`을 정확한 preflight source commit으로 fast-forward한
뒤 원본 config로 실행해야 한다.
