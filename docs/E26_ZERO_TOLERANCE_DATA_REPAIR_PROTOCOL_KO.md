# E26 Data Repair R1: zero-tolerance near-duplicate protocol

## 목적과 원본 보존

Stage-2의 `BLOCKED_DATA_SOURCE`와
`FAIL_PENDING_MANUAL_AUDIT` 판정은 변경하지 않는다. R1은 그 판정을
재평가하는 실험이 아니라, model outcome을 보기 전에 추가한 별도 prospective
data-admission protocol이다.

R1의 정책은 단 하나다.

```text
frozen detector가 protected–train pair를 flag
→ general_train 쪽 normalized-content SHA를 무조건 제외
```

사람 또는 모델의 semantic label, 추천 label, reward, GRPO, validation loss,
transaction score는 사용하지 않는다. `tokenizer_only`, `general_validation`,
`general_test` 문서는 삭제하지 않는다.

## 고정된 입력과 detector

- 원본 corrected audit: 541 pair, 파일 SHA
  `9bfe3c84df9368955b97ec6285db01450dff912cecd9f2f3334a7b8ced18f17a`
- FineWeb-Edu revision:
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- 선택 shard: `0, 2, 4, 9, 13`
- text identity: NFC/LF-normalized UTF-8의 SHA-256
- near duplicate: 5-word shingles, 128 permutations, 32×4 bands,
  threshold 0.80, seed 260026
- protected index + streamed train의 asymmetric comparison

Detector 수치와 split은 바꾸지 않는다. 원본 tokenizer도 다시 학습하지 않는다.

## 결정적 exclusion과 backfill

1. 541 pair의 모든 `general_train` content SHA를 중복 제거해 초기 exclusion
   set을 만든다.
2. 동일 pinned Parquet byte를 canonical shard/row 순서로 다시 읽고 exact
   content dedup을 재실행한다.
3. Exclusion set의 train 문서만 제거한다.
4. Train memmap selection은 V1과 동일하게 `content_sha256` 오름차순이다.
5. Whole-document policy로 400M token을 채우며, 빠진 용량은 같은 정렬의 다음
   eligible 문서로만 보충한다.
6. Protected memmap은 V1과 byte-identical해야 한다.
7. 전체 eligible index에 동일 corrected audit를 다시 실행한다.
8. 새 flag가 있으면 모든 train-side SHA를 exclusion set에 단조롭게 추가하고
   fresh iteration namespace에서 2–7을 반복한다.

Shard/row source 순서로 train을 새로 표본화하거나 새로운 shard를 선택하는 것은
금지한다. 기존 V1 train selection law는 content-hash 순서였기 때문이다.

## 종료와 claim 경계

허용되는 terminal disposition은 다음뿐이다.

```text
ZERO_PROTECTED_TRAIN_FLAGS
BLOCKED_SOURCE_CAPACITY
BLOCKED_PROVENANCE
```

Zero flag일 때만 `scientific_data_readiness_v3`를 생성한다. 이 receipt는
scientific input provenance만 열며 LM 결과를 뜻하지 않는다. 이 단계에서는
GPU resource preflight, E26a, E26b, E26c를 실행하지 않는다.
