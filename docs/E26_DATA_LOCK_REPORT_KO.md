# E26 data lock 결과

## 판정

```text
execution_status: COMPLETED_WITH_REGISTERED_BLOCK
stage2_disposition: BLOCKED_DATA_SOURCE
scientific_input_eligible: false
reason: PREREGISTERED_NEAR_DUPLICATE_GATE_FLAGGED_541_PAIRS
human_adjudication_pending: true
```

FineWeb-Edu source 고정, exact content deduplication, 16,384-token BPE
독립 replay, shared `uint16` memmap, transaction replay와 paired schedule은
완료됐다. 그러나 사전등록한 5-word MinHash/32×4 LSH audit가 protected
split과 train 사이의 후보 541쌍을 검출했으므로 전체 data bundle은
scientific E26a 입력으로 승격하지 않았다.

| 구성 | 결과 | 정확한 artifact |
|---|---:|---|
| FineWeb-Edu pin | PASS | revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, shards `0,2,4,9,13` |
| Exact dedup/split | PASS | 3,093,101 seen, 3,053,890 unique, 39,211 duplicates |
| 16K BPE replay | PASS | vocab 16,384, independent artifact sets identical |
| General memmap | PASS | train 400,005,587; validation 5,001,316; test 5,000,215 tokens |
| Transaction replay | PASS | 6,000 episodes, exact replay, visible leakage 0 |
| Paired schedule | PASS | deterministic bounded 80/20 token mix |
| Near-duplicate gate | **FAIL** | 541 pairs, `FAIL_PENDING_MANUAL_AUDIT` |

주요 파일 SHA-256:

| 파일 | SHA-256 |
|---|---|
| Source inventory | `990bb95b7ed09913bd22e6a2d81cc42ed785edb86a4163060d8d43166e825753` |
| Download receipt | `ed0044dfcdac39e864130b3caa2285c254931925ca69a4b67ffb599811fbb4a6` |
| Dedup receipt | `8b69579d93e76b0184dec5b0b3cac818384e1c1627378af86eae2a1c439e6dca` |
| Tokenizer manifest | `9822e13d8cbbef4c3f3fac4d03674dcde4f6454832a05e23559ed43532cef40d` |
| Tokenizer replay | `9056cc3b79241a3f54c30b6a6303c4181ce45753fcd7ba08e01a3ee030037de8` |
| General memmaps | `c6842c9ed5bca87ce305e9784b5fb45c35826a1defb8dc8556c0f012770ff5e6` |
| Transaction replay | `71849b23f58da7fbaf4c1e835e7f33cb2b5f12d03a8c038a445dbe51d51449e6` |
| Paired schedule | `19d04449aa5fd0c1f5f675fe3768ca7ceef98c79908b6d5d4c0ceeeaf735b8f5` |

## Near-duplicate 재현과 수동 검수

동일한 corrected audit를 `catena-v6`와 pinned isolated data environment에서
각각 실행했다. 실행시간은 2,338.76초와 2,323.86초였고 두 결과 파일은
byte-identical했다.

```text
audit file SHA-256: 9bfe3c84df9368955b97ec6285db01450dff912cecd9f2f3334a7b8ced18f17a
internal audit SHA: 37e5616ab414c9e018ef46dcf8b6d77b927de222ba38f1e86ac7ffdfd8c5fb35
test–train: 75
validation–train: 82
tokenizer–train: 384
estimated Jaccard range: 0.8046875–1.0
```

원본 human-review package v1은
`/data/minjun_dev/CATENA/e26_data_v1/near_duplicate/reviewer_package_v1_expansion1/`
에 그대로 보존했다. 사람이 읽기 어려운 CSV의 JSON-string wrapping만 제거한
clean v2는
`/data/minjun_dev/CATENA/e26_data_v1/near_duplicate/reviewer_package_v2_expansion1_clean/`
에 있다. v2 JSONL은 v1과 byte-identical
(`1d59d4ba551715995f5739926295d66d4a30b290f3974de70e5a6e0b3cb801e7`)이고
clean CSV SHA-256은
`b023c0ac46cbf8f4eb04398e0fc3ae130a993bab27cdaf8fbd553da28058b3ac`다.
두 형식 모두 541행이며 label과 note는 전부 비어 있다. Codex는 어떤 pair도
판정·삭제하지 않았고 threshold, split, seed도 바꾸지 않았다.

## Claim 경계와 다음 dependency

현재 허용되는 결론은 “데이터 구축 구현과 독립 replay는 재현되었으나
near-duplicate data gate가 열리지 않았다”뿐이다. LM 성능, E26a numerical
readiness, candidate/budget 또는 E26b GO는 주장할 수 없다. 독립 인간 검수가
끝난 뒤에도 원본 audit는 그대로 보존하며, 필요하면 새 namespace에서
prospective data repair와 전체 hash 재생성부터 수행해야 한다.
