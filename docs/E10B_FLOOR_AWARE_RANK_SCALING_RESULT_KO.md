# E10b Floor-Aware Learned Rank Scaling 결과

## 판정

```text
execution_status: PASS
original_e10_claim_status: NOT_OPENED
prospective_e10b_status: SUPPORTED
evidence_tier: CONTROLLED_REFERENCE
```

E10b main run은 `20260727T190906.272784Z`다. 원본 E10 report와 strict
monotonic gate는 수정하지 않았다. E10b는 동결한 240개 checkpoint를
재학습하거나 선택하지 않고, 완전히 새로운 descriptor test namespace에서
평가했다.

## 주요 결과

| 지표 | 결과 | 판정 |
|---|---:|---|
| Frozen checkpoint hash | 240/240 | PASS |
| Fresh evaluation rows | 240/240 | PASS |
| Pre-saturation adjacent pairs | 80 | 모두 식별 가능 |
| Pre-saturation non-increasing pairs | 80/80 | PASS |
| Saturated pairs | 120 | 기록 후 monotonic denominator에서 제외 |
| Minimum-rank tracking | 40/40, 1.000 | PASS |
| Minimum rank = intrinsic rank | 40/40, 1.000 | PASS |
| Seedwise nondecreasing | 8/8 | PASS |
| Seed-level sign-flip | `p=0.00390625` | PASS |

새 numerical tolerance를 도입하지 않았다. Lower-rank의 fresh-set
exact-target recovery가 기존 threshold `0.95` 미만일 때만 다음 rank가
raw error를 낮춰야 한다. Lower-rank가 이미 threshold를 만족하면 그 pair는
post-saturation diagnostic으로 남기되 scaling gate의 분모에서 제외했다.

## 해석

등록된 smooth synthetic operator family에서는 exact target을 복구하는 데
필요한 최소 learned control rank가 demand intrinsic rank와 정확히
일치했다. 원본 E10의 full gate failure는 rank-tracking 실패가 아니라,
이미 numerical floor에 도달한 이후의 비식별 미세 증감을 strict
monotonic violation으로 센 protocol 문제였다.

Pretrained LM, natural-language transaction 또는 official backend에 대한
일반 rank law는 열리지 않는다.

## Immutable provenance

| 항목 | SHA-256 |
|---|---|
| Report | `30f2f781bcc8528964602e0c66e1b61bb9d71a6ca5f964b833b2551c93b72484` |
| Run manifest | `758580489f2e88969fe03d1c5841735feeba1294107d1f36976ad125adf7bd83` |
| Fresh metrics | `9238bac0a4ae59d5fe0ab24c28d2c9458f5e4cabc8636c2f483b472fdd1c991c` |
| Pre-saturation pairs | `2c7706319c9a70211edc12e365ef2e43b83e5efdbefe38973e75479da2c2545a` |
| Seed effects | `5716a4b91b89f4a77ef24db61f8c5323e4e1cd2f75509b374301ce8f0cf88bb0` |
| Checkpoint verification | `64417c44284f3887b53a1b2bf385c7bbfeac529c402f4a0cfea6dae433c7d023` |
| Source freeze | `de749a4c087343489e94f0bba04bb4ee9b8d690c6691cf2df398e2fb6cf85650` |
| Protocol lock | `97dde4cca8edef545415a22edbe3d80ed5261a2894c9e8afc5b0123e713cfee4` |
