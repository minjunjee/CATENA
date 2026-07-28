# Post-core dependency DAG

```text
기존 immutable evidence
H1 ─ H2/E02b ─ H4
 └──── H3/E03b
       H5 NO-GO (closed)

즉시 병렬 실행
├── E10 learned-rank scaling ───────────────┐
├── E11 representation-control coadaptation ├─> E12/E13 해석 강화
├── E12 control-algebra lattice ────────────┘
├── E13a pilot (immutable, dependency 불가)
├── E13a-R1 paired sequence calibration ── GO ─> E13b per-run ─> E13c aggregate ─> E14 plan continuation
├── E15 official backend gate (별도 환경)
└── E16 core evidence freeze
```

## Dependency가 없는 실험

E10, E11, E12, E13a pilot, E13a-R1, E15 dry-run, E16은 서로
독립적이다. 기존 H1-H4 결과를 설계 근거로 사용하지만 checkpoint
dependency는 없다.

## Hard dependency

- E13b는 prospective E13a-R1 GO가 필요하다. 원본 E13a pilot의 GO는
  dependency로 사용할 수 없다.
- E13c는 tied/dual, 5-seed E13b의 완전한 update×gap grid와 유일한
  eligible run provenance가 필요하다. 5 seeds는 one-sided exact sign-flip
  `alpha=0.05`의 최소 식별 설계다 (`p_min=1/32`).
- E14는 latest repaired E13c-R1 `MAIN/PASS/SUPPORTED` report가 봉인한
  정확한 5개 dual E13b-R1 checkpoint가 필요하다. 임의 checkpoint나 glob
  선택은 금지한다.
- Official-backend claim은 E15 PASS가 필요하다.

## 논문 claim dependency

- REALM core: H1-H4만으로 완결된다.
- E10/E11/E12: post-core theory/architecture extension. 성공 시 full-paper claim을 강화하지만 REALM core를 재판정하지 않는다.
- E13c: sequence bridge. 성공해도 language-model claim이 아니라 structured event-sequence claim이다.
- E15 이후의 replication만 official architecture claim을 연다.
