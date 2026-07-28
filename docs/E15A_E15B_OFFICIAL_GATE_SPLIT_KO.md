# E15a/E15b official operator gate 분리

## 목적

기존 E15의 GDN2/KDA와 KVEraser 동시 dependency를 분리한다. 어느 한쪽이
미구성이라는 이유로 다른 backend의 준비 상태를 숨기지 않으며,
reference/mock fallback은 두 gate 모두 금지한다.

## E15a — Official GDN2/KDA

정확한 upstream full commit SHA를 고정하고 다음을 모두 통과해야 한다.

1. FP32 full-sequence 대 chunked recurrence parity
2. GDN2 tied reduction 대 KDA FP32 parity
3. BF16 parity
4. backward gradient finite
5. state carry·clone·restore
6. erase/write intervention hook confinement

하나라도 누락되거나 실패하면 official operator claim은 열리지 않는다.

## E15b — Official KVEraser

별도 repository와 commit에서 full-recompute agreement, finite backward,
state carry·clone·restore를 검사한다. E15a와 환경·artifact를 공유하지
않는다.

## 상태 의미

| 상태 | 의미 |
|---|---|
| `DRY_RUN` | Gate와 artifact 경로만 확인; scientific evidence 아님 |
| `NOT_CONFIGURED` | Repository, full commit 또는 plugin 미제공 |
| `FAIL` | 구성된 official backend가 등록 check를 통과하지 못함 |
| `PASS` | Pinned revision과 모든 check 통과; 해당 operator claim만 eligible |

실제 실행은 기존 `catena-v6`을 오염시키지 않는 별도 environment 또는
container에서 수행한다.

