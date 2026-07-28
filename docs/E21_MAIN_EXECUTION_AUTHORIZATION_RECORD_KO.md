# E21 MAIN 실행 권한 기록

- 기록 시각: `2026-07-28T07:13:00Z`
- E21 final lock SHA-256:
  `e07139064b6f2cf1ca990f4f595d38c64f295cd7b25ef2fd3a935cbefe498579`
- 첫 MAIN seed: `113`
- 첫 MAIN run:
  `e21a_structured_sequence_localization_transfer/20260728T070300.327708Z`

사용자는 post-core 실험을 직접 실행하고, 즉시 실행 가능한 작업이
없어질 때까지 진행하며, minor implementation bug는 자율 수리하라고
명시했다. 또한 4-GPU 실행과 learned address/candidate transfer를 후속
우선순위에 포함했다. 이에 따라 E21 lock의
`main_execution_requires_explicit_user_instruction=true` 조건은 충족된
것으로 기록한다.

이 문서는 실행 권한과 운영 시점을 기록하는 additive 문서다. E21
config, source protocol, metric, threshold, seed, model 또는 artifact를
변경하지 않는다. E21은 fixed structured identifier codebook과 explicit
demand algebra의 `CONTROLLED_REFERENCE` 실험이며 H5, 자연어, novel
identifier, pretrained/official model 또는 agent claim을 열지 않는다.
