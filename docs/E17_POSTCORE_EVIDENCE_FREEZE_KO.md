# E17 — Post-Core Evidence Freeze

## 목적

E17은 완료된 E10–E15 artifact를 수정하지 않고, post-core 논문에 사용할
수 있는 exact run과 claim 경계를 하나의 registry로 검증·동결한다.
`latest.json`이나 디렉터리 시간순 선택을 사용하지 않으며, config와
evaluator에 고정된 experiment ID와 run ID가 정확히 일치해야 한다.

E16의 H1–H5 core registry는 수정하거나 대체하지 않는다. E17은 E16의
hash-pinned evidence utility를 재사용하는 별도 post-core registry다.

## 고정 record

| Record | Exact run | Disposition | 역할 |
|---|---|---|---|
| E10 original | `e10_learned_rank_scaling/20260727T184326.484361Z` | `NOT_OPENED` | 원본 gate 보존 |
| E10b | `e10b_floor_aware_rank_scaling/20260727T190906.272784Z` | `SUPPORTED` | Prospective repair |
| E11 original | `e11_representation_control_coadaptation/20260727T180703.763554Z` | `NOT_OPENED_SCALE_RESTRICTION` | 원본 gate 보존 |
| E11b | `e11b_scale_normalized_coadaptation/20260727T183004.928280Z` | `SUPPORTED` | Prospective repair |
| E12 | `e12_control_algebra_lattice/20260727T184511.437394Z` | `SUPPORTED` | Artifact-complete canonical run |
| E13a original | `e13a_sequence_floor_throughput/20260727T180703.836996Z` | `CALIBRATION_PILOT_ONLY` | Main dependency 아님 |
| E13a-R1 | `e13a_r1_sequence_floor_throughput/20260727T183609.755945Z` | `GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY` | Repaired E13b-R1 dependency 아님 |
| E13a-R2 | `e13a_r2_sequence_floor_throughput/20260727T190642.222102Z` | `GO_FOR_E13B_R1_CALIBRATION_ONLY` | Repaired pipeline의 calibration dependency |
| E13c-R1 | `e13c_r1_transactional_sequence_aggregate/20260727T214126.954177Z` | `SUPPORTED` | Sealed structured-sequence aggregate |
| E14 | `e14_plan_continuation/20260727T214143.455051Z` | `SUPPORTED_STRUCTURED_PROXY_ONLY` | Sealed structured continuation proxy |
| E15 | `e15_official_backend_gate/20260727T184517.578907Z` | `NOT_CONFIGURED_DRY_GATE` | Canonical dry gate |

E12의 첫 corrected run `20260727T182449.721061Z`는 checkpoint artifact
contract가 불완전하므로 E17 record가 될 수 없다. E13b 개별 run은
E13c-R1 freeze 내부의 source provenance로 봉인되며, 독립 claim record로
중복 등록하지 않는다.

## 검증 항목

각 record에서 다음을 모두 검증한다.

1. Code-level protocol과 config의 exact experiment ID, run ID, role,
   disposition 일치
2. `report.json`과 `run_manifest.json` SHA-256
3. Report와 manifest의 등록된 핵심 field
4. 해당 experiment의 핵심 artifact freeze SHA-256과 disposition
5. E13a original의 repository calibration lock
6. 다음 scope flag 다섯 개

```text
controlled_claim_eligible
structured_sequence_claim_eligible
official_operator_claim_eligible
language_model_claim_eligible
agent_claim_eligible
```

`structured_sequence_claim_eligible=true`인 record는 E13c-R1 하나뿐이다.
Official operator, language model과 agent flag는 모든 현재 record에서
`false`다. E14는 structured synthetic proxy이므로 일반 agent 또는
independent plan-semantics evidence가 아니다.

## 출력

E17 run은 다음을 생성한다.

- `postcore_evidence_registry.json`
- `postcore_results_macros.tex`
- `report.json`
- schema-v2 `run_manifest.json`

Dry-run도 실제 exact path와 모든 선언 hash를 읽어 검증한다. 다만 registry
mode를 `VALIDATED_DRY_RUN_NO_CANONICAL_FREEZE`로 기록하고
`canonical_freeze_written=false`를 유지한다.

## 실행

CPU validation dry-run:

```bash
python experiments/e17_postcore_evidence_freeze.py \
  --config configs/e17_postcore_evidence_freeze.yaml \
  --device cpu \
  --artifact-root /tmp/catena_e17_validation \
  --evidence-root /data/minjun_dev/CATENA/artifacts \
  --dry-run
```

`--evidence-root`는 기존 source artifact를 읽기만 하며,
`--artifact-root`는 E17 dry-run 출력만 받는다. 따라서 위 명령은 기존
artifact tree에 E17 directory나 `latest.json`을 추가하지 않는다.

Main 실행:

```bash
python experiments/e17_postcore_evidence_freeze.py \
  --config configs/e17_postcore_evidence_freeze.yaml \
  --device cpu \
  --evidence-root /data/minjun_dev/CATENA/artifacts \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

Main은 별도 승인 후에만 실행한다. 구현 검증 단계에서는 focused test와
temporary mirror를 사용한 CPU dry-run만 수행하며 canonical E17 main
artifact를 만들지 않는다.

## Claim 경계

E17은 새 과학적 결과를 만들거나 원본 판정을 변경하지 않는다. E10과
E11의 원본은 계속 열리지 않으며 E10b와 E11b만 prospective repair
claim을 가진다. E13a-R2는 sequence main의 calibration dependency일 뿐
반복 update claim 자체가 아니다. E15 dry gate는 official backend
evidence가 아니며, reference 결과를 official operator, language model,
natural-language transaction 또는 agent 결과로 승격할 수 없다.
