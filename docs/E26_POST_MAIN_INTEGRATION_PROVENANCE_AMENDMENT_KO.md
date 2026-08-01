# E26 post-main integration provenance amendment

## 배경

사용자 승인에 따라 `exp/e26-autoregressive-lm`을 2026-08-01에 로컬 및
원격 `main`으로 fast-forward 통합했다. 기존 Stage-2 frozen receipt는 live
repository HEAD가 pre-E26 commit
`adfdeaf9e87a8602a8e334915d87acb9ff25af39`와 정확히 같아야 한다고
가정했으므로, additive 통합만으로도 재검증이 실패했다.

## 수리한 계약

Pre-E26 source invariance는 다음을 모두 요구한다.

1. expected pre-E26 commit이 현재 live HEAD의 ancestor다.
2. expected commit에 tracked된 556개 base blob이 현재 worktree에서
   byte-identical하다.
3. base file count와 aggregate SHA-256이 등록값과 일치한다.
4. live worktree가 clean하다.

현재 HEAD가 expected commit과 같은지는 diagnostic `head_matches`로 계속
기록하지만, additive descendant 자체를 failure로 취급하지 않는다. Expected
commit이 ancestry에서 사라지거나 기존 base blob 하나라도 바뀌면 계속
fail-closed한다.

## 과학적 경계

```text
amendment_type: POST_INTEGRATION_PROVENANCE_ONLY
data_rows_changed: false
tokenizer_changed: false
threshold_or_metric_changed: false
artifact_changed: false
scientific_e26a_started: false
```

이 amendment는 기존 Stage-2 preflight 결과를 재판정하지 않는다. Main 통합
이후 생성되는 새 frozen receipt만 descendant-aware contract로 검증한다.
