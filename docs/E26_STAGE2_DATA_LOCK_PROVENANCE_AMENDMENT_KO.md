# E26 Stage-2 data-lock provenance amendment

## 범위

Near-duplicate audit가 실행 중이던 동안
`configs/e26_data_lock_v1.yaml`에 누락되어 있던 사전 E26 source/artifact
불변성 수치 세 개를 추가했다.

```text
repository.pre_e26_source_file_count
repository.pre_e26_source_aggregate_sha256
repository.frozen_artifact_file_count
```

변경 전 파일 SHA-256은
`360ad972c41cf10eaed4806b304c577c12df760b9921b1488216cc18221a69c0`,
변경 후 SHA-256은
`4b80aa7c620d51eca8f59f4d06756daa0006ed3ae62b1b61420d2be4543c25e2`다.

## 과학적 영향

이 변경은 source/artifact 불변성 provenance와 fail-closed admission만 보강한다. Near-duplicate 프로그램은 실행 중
YAML을 읽지 않았고, 다음 항목은 바뀌지 않았다.

- document SQLite index와 SHA-256
- audit Python source와 SHA-256
- normalization, shingle, MinHash, LSH algorithm
- seed, threshold, band/row 수
- content-hash split 및 문서 행

두 독립 환경의 corrected 32×4 LSH 결과는 byte-identical했다. 따라서 이
amendment는 이미 실행된 audit의 계산을 변경하지 않는다. 다만 541개 후보가
검출되었으므로, 이 사실과 무관하게 현재 data bundle은 human adjudication
전까지 scientific input으로 사용할 수 없다.

## 판정

```text
amendment_type: PROVENANCE_ONLY_DESCENDANT
algorithm_or_threshold_changed: false
scientific_input_eligible: false
blocking_reason: 541_NEAR_DUPLICATE_FLAGS_PENDING_HUMAN_AUDIT
```
