# E05a human-audit adjudication implementation lock

- Frozen at: `2026-07-27T07:32:48.652814Z`
- Parent E05a/E05b protocol SHA-256:
  `e235a351ca84589d92a211e78f6f4ebe6631ce228adad8f8da34dec17527b8e0`
- Parent protocol lock SHA-256:
  `4aaadf16eb02abff19f519191c0c56de0ad383cf01dfeea91fedacfb8a41980e`
- Config:
  `configs/e05a_semantic_audit_adjudication.yaml`
- Config canonical SHA-256:
  `c564a2058d7647ebdf2eb38b022622fc0c6b47f4754dd3cb57c951fa16cb98b6`
- Config byte SHA-256:
  `c43e5cd6914b22006bd5ca9195504d4cec7d6c6478c9eb8efba111f04f82eb7c`
- Audit item/model outcome inspected before lock: `false`
- Human review file present before lock: `false`

이 stage는 E05a의 locked audit-item artifact를 수정하지 않는다. Reviewer A,
Reviewer B, adjudication 파일을 별도 immutable run directory에 복사하고 hash를
기록한다. 두 reviewer는 사람이어야 하며 AI agent 판정을 human review로
기록하지 않는다.

Pass는 E05b training dependency일 뿐 H5 support가 아니다. 의미 보존
`>=.95`, leakage `<=.02`, 두 label 각각 raw agreement `>=.80`, 300/300
adjudication을 모두 요구한다.
