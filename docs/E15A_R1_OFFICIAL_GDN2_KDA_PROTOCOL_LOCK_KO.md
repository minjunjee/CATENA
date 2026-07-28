# E15a-R1 official GDN2/KDA compatibility repair lock

동결 시각: `2026-07-28T06:21:47.075395Z`

## 원본 disposition

원본 E15a run `20260728T061317.967602Z`는 `FAIL`로 동결한다. GDN2
operator가 `chunk_gla_fwd_o_gk(..., use_exp2=True,
transpose_state_layout=...)`를 호출했지만, 최초 고정한 FLA commit
`19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5`는 이미 해당 API를
변경한 뒤여서 어떤 scientific gate도 평가되지 않았다.

```text
original_status: FAIL
original_claim_open: false
failure_stage: PINNED_DEPENDENCY_API_COMPATIBILITY
evaluable_scientific_gates: 0 / 6
```

## Prospective repair

E15a-R1은 GDN2 source, probe, metric, tolerance와 여섯 gate를 변경하지
않는다. FLA main history에서 GDN2가 요구하는 `use_exp2`와
`transpose_state_layout` 인자를 동시에 제공하는 마지막 revision,
`4b02d15d6a68700181b180235be62a9fb95d2a38`을 새 clean detached
worktree로 고정한다.

| 항목 | R1 고정값 |
|---|---|
| GDN2 commit | `95709fc250357c2dd109361c353192f2aa5913f9` |
| FLA remote | `https://github.com/sustcsonglin/flash-linear-attention.git` |
| FLA commit | `4b02d15d6a68700181b180235be62a9fb95d2a38` |
| FLA checkout | `/data/minjun_dev/CATENA/official_sources/flash-linear-attention_4b02d15_clean` |
| Separate environment | `/data/minjun_dev/CATENA/envs/gdn2_official_95709fc` |

이 revision 선택에는 E15a scientific metric을 사용하지 않았다. API
signature와 Git ancestry만 사용했으며, E15a-R1 MAIN 전에 config·plugin
및 source hash를 고정했다.

## 변경하지 않은 gate

1. Full sequence 대 carried chunks FP32 relative L2 `<= 1e-5`
2. Tied GDN2 reduction 대 KDA FP32 relative L2 `<= 1e-5`
3. BF16 대 FP32 relative L2 `<= 5e-3`
4. 모든 등록 backward gradient finite
5. State carry·clone·restore
6. Erase/write intervention hook confinement과 causal-prefix 보존

R1이 PASS해도 official operator parity evidence만 열린다. E02b/E12/E13
official replication, pretrained LM, language 또는 agent claim은 별도
실험 전까지 닫혀 있다.
