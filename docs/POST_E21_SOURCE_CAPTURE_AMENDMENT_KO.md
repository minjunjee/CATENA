# Post-E21 Git Source Capture Amendment

## 목적

Post-E21 MAIN을 시작한 source tree의 fingerprint는 다음과 같다.

```text
Git commit:
51156242dfc429cb66d577c144b8d38a5ae38551

Source-tree fingerprint:
9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac

Fingerprint file count:
472
```

이 fingerprint는 Git ignore 여부와 관계없이 repository 아래의 source
파일을 내용과 상대 경로로 해시한다. 감사 과정에서 이 tree에 포함되지만
Git commit에는 들어 있지 않은 `src/catena/data/*.py` 23개가 발견됐다.
원인은 `.gitignore`의 광범위한 `src/catena/data/*` 규칙이다. Clean-worktree
회귀에서는 paper scaffold가 요구하는
`papers/transactional_control_algebra_long/data/source_manifest.json`도
광범위한 `data/` 규칙 때문에 누락된 것을 추가로 확인했다. JSON은 현재
source-tree fingerprint suffix 집합에는 포함되지 않지만 paper build의
명시적 재현성 dependency다.

이 amendment는 해당 24개 파일의 기존 바이트를 그대로 Git에 포함해
후속 E22b/E23b source-lock tag가 실제 실행 source를 재구성할 수 있게 한다.
다음을 명시적으로 금지한다.

- 기존 24개 파일의 내용 변경
- E00--E22a artifact, report, checkpoint 변경
- config, seed, threshold, metric, model, precision, data namespace 변경
- 완료된 E22a의 재판정

## 검증 계약

1. 24개 파일 각각의 SHA-256은
   `docs/POST_E21_SOURCE_CAPTURE_LOCK.json`과 일치해야 한다.
2. 24개 파일의 정렬된 `SHA-256  relative/path` manifest SHA-256은
   `d03a3bb0a74ba99fb1c9abd1f0fa0d982b90ead7f8c841e90e8b91b172f62ea0`
   이어야 한다.
3. 원 source-tree fingerprint는 E22a 완료 시점까지
   `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac`
   (472 files)로 유지되어야 한다.
4. 이 amendment와 seed-sharding 파일을 통합한 뒤에는 새 clean commit과
   annotated source-lock tag를 만들고, E22b/E23b equivalence proof와 MAIN은
   그 새 source fingerprint에만 결속한다.

이 변경은 source capture와 실행 provenance만 보강한다. Scientific
protocol amendment가 아니며 새로운 claim을 열지 않는다.
