# E26 Stage-2 backend provenance DAG

Backend provenance는 다음의 비순환 DAG로 고정한다.

```text
clean execution source + E26a config
        │
        └── backend_candidate_lock.json
              capabilities: all false
              protocol-bound upstream
                    │
                    ├── candidate-matrix numerical audit
                    └── packed-cursor/restart audit
                              │
                              └── backend_preflight_manifest.json
                                    e26a_candidate_capable: audit 전체 PASS일 때만 true
                                    e26a_gate_capable: false
                                    scientific_main_capable: false
                                    parity_verified: false
```

`backend_candidate_lock.json`은 protocol lock보다 먼저 생성한다. Candidate id와
raw config hash, clean Git commit, result Markdown를 제외한 execution-source
inventory 및 compiled backend policy를 고정하며 overwrite하지 않는다.
자기 자신이 source inventory에 들어가는 순환을 피하기 위해 이 lock은 worktree
밖의 immutable protocol-input 디렉터리에 기록한다.

검증 시 lock의 `source_commit`은 현재 `HEAD`와 exact equality를 요구하지 않고,
존재하는 full commit이며 현재 `HEAD`의 ancestor임을 요구한다. 이후 결과
Markdown만 추가한 descendant commit을 허용하기 위해서다. 반면 Markdown을
제외한 execution-source inventory, config SHA와 candidate config SHA는 현재
실행 입력과 byte-exact하게 같아야 하므로 executable drift는 계속 fail-closed다.

Numerical/restart receipt는 candidate lock SHA와 protocol 및 data upstream hash를
고정한다. 자기 자신이나 downstream promotion manifest의 hash는 포함하지 않는다.

`backend_preflight_manifest.json`은 두 audit receipt와 candidate lock을 파일
SHA-256 및 embedded canonical receipt SHA로 묶고, exact torch/CUDA/driver/GPU
inventory와 candidate별 compiled diagnostics를 기록한다. 모든 candidate,
두 variant, FP32/BF16, zero/prefilled arbitrary partitions, gradient accumulation,
packed-cursor replay 및 new-process restart가 통과하고 fallback/graph break가
0일 때만 `e26a_candidate_capable=true`가 된다.

이 promotion은 canonical scientific E26a 실행 승인이 아니다. E26a gate,
scientific MAIN 및 parity claim은 계속 닫혀 있다.

Backend promotion 이후의 target-context throughput 측정은 별도
`resource_preflight.json` downstream node로 기록한다. Protocol의 upstream hash
DAG에는 이 receipt를 역방향으로 추가하지 않는다. Canonical admission은 explicit
receipt path와 사용자가 승인한 exact file SHA-256을 별도 CLI 입력으로 요구하고,
resource receipt가 선택한 candidate id와 token budget을 고정한다. 자세한 계약은
`docs/E26_STAGE2_RESOURCE_PREFLIGHT_LOCK_KO.md`를 따른다.
