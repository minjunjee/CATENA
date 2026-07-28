# E02b prospective absolute-equivalence protocol amendment

동결 시각: `2026-07-26T17:11:04.190032Z`

상태: **FROZEN BEFORE ANY E02b DRY-RUN OR APPLICATION EVALUATION**

이 amendment는 기존 E02를 재판정하지 않는다. 기존 E02의 report, metric,
checkpoint와 claim status는 그대로 보존하며, E02b만 별도의 prospective
실험으로 판정한다.

## 1. 기존 E02의 불변 판정

```text
source_run: 20260726T153504.455509Z
execution_status: PASS
original_confirmatory_status: INCONCLUSIVE
reason: PREREGISTERED_SYMMETRIC_RELATIVE_GATE_UNIDENTIFIABLE
evaluable_gates_passed: 5/5
original_h2_claim_open: false
```

| Source artifact | SHA-256 |
|---|---|
| `run_manifest.json` | `1ea4ad867ffbc86bca3f4ee8f3eceb698089f973db250920a0eeb3dc39641d4c` |
| `report.json` | `f3df03e231598d6eda11ebf71825ab418cc9a59ac9a96a299caff617291e4211` |
| `episode_metrics.jsonl` | `9a1f85d4c18366f062caceec47b83d16763dffcbdeaef305b3ecc2010878c5fd` |

SUPERSEDE relative-equivalence gate는 tied와 dual 모두 symmetric target을
수치적으로 정확히 실현해 oracle headroom이 사라지므로 식별되지 않았다. 이
결함은 E02의 `INCONCLUSIVE` 판정을 유지한 채 E02b에서만 수리한다.

## 2. Prospective repair와 데이터 계약

SUPERSEDE에는 PRESERVE에서 이미 사용한 raw-MSE absolute-equivalence margin
`±0.0005`를 적용한다. 원 E02 test row는 재사용하지 않는다.

E02b는 사용자가 지정한 더 엄격한 **frozen-controller unseen norm/angle OOD
heldout extension**이다. 따라서 같은 geometry cell의 단순 protocol repair가
아니며, 결과는 `prospective absolute-SUPERSEDE repair plus OOD geometry
extension`으로만 해석한다.

| 항목 | 동결값 |
|---|---:|
| Frozen strict checkpoint pairs | 8 tied/dual pairs |
| Operations | PRESERVE, ADD, INVALIDATE, SUPERSEDE |
| Episodes / operation / checkpoint | 512 |
| Main total rows | 16,384 |
| Norm pairs | `(0.75,0.90)`, `(0.90,1.25)`, `(1.10,0.80)`, `(1.25,1.10)` |
| Old/new angles | `45°`, `75°`, `105°`, `135°` |
| Geometry cells | 16, each repeated 32 times per operation/checkpoint |
| Main relative seed range | `[62500,64547]` per checkpoint seed block |
| Dry relative seed range | `[90000,90063]` for one checkpoint |
| Candidate/address | OracleCandidate / fixed oracle address |

Main, dry, E02 train/validation/test와 reserved E04 seed range는 서로 겹치지
않는다. Main 실행에는 ADD와 INVALIDATE가 checkpoint seed마다 각각 512개 모두
normalized-gain eligible이어야 한다. 이 registered support가 부족하면 gate
실패가 아니라 `INCONCLUSIVE / REGISTERED_ASYMMETRIC_SUPPORT_INCOMPLETE`로
판정한다.

## 3. 동결된 여섯 조건

| Gate | Freshness | 판정 규칙 |
|---|---|---|
| Asymmetric normalized gain | fresh | CI lower `> 0.10` 및 8-seed exact sign-flip `p≤0.05` |
| PRESERVE raw equivalence | fresh | fixed-checkpoint episode-bootstrap CI가 `±0.0005` 내부 |
| SUPERSEDE raw equivalence | fresh | fixed-checkpoint episode-bootstrap CI가 `±0.0005` 내부 |
| Asymmetric−symmetric interaction | fresh | CI lower `>0` 및 8-seed exact sign-flip `p≤0.05` |
| Retention non-inferiority | fresh | CI upper `≤0.0005` 및 8-seed exact sign-flip `p≤0.05` |
| Tuning direction | inherited | 원 E02 report의 preregistered 8/8 positive fact; tuned checkpoint replay 없음 |

따라서 E02b support는 **fresh gate 5개와 inherited fact 1개**의 conjunction이다.
원 E02에 저장된 tuned checkpoint pair가 없으므로 tuned state를 fresh set에서
재평가했다고 주장하지 않는다.

## 4. No-retraining 및 evidence 계약

1. 기존 E02는 계속 `INCONCLUSIVE`다.
2. relative SUPERSEDE gate의 구조적 비식별성을 별도로 기록한다.
3. 새 absolute margin은 기존 PRESERVE raw margin과 동일하다.
4. 원 E02 test row는 재사용하지 않는다.
5. checkpoint를 재학습하거나 E02b 결과로 선택하지 않는다.
6. 새 seed와 16개 unseen geometry cell만 평가한다.
7. 이 amendment는 E02b dry-run과 application evaluation 전에 동결됐다.

```text
evidence_tier: CONTROLLED_REFERENCE
scientific_evidence: false
official_backend_claim_eligible: false
language_model_claim_eligible: false
architecture_transfer_claim_eligible: false
```

## 5. Config lock

| Config digest | SHA-256 |
|---|---|
| Canonical payload | `9b7d299a916003a9d4b5038a8db325d4609d618b3d9919ba7b603cdf8d76f781` |
| File bytes | `4e269572652f494977f0370a82c5104bd083ae4e7a4f570635d4685af15fe956` |

이 문서의 SHA-256은 별도 lock 문서와 E02b runner에서 검증한다. 그 뒤 최종
source tree 전체를 새 E00 run으로 다시 고정한다.
