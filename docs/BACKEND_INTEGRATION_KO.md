# 공식 Backend 통합 규약

## GDN2/KDA

1. NVlabs official repository를 별도 checkout한다.
2. exact commit SHA를 config와 run manifest에 기록한다.
3. FP32 reference full/chunk parity, BF16 accelerated parity, tied-GDN2/KDA equivalence를 먼저 통과한다.
4. community checkpoint는 official checkpoint로 표기하지 않는다. provenance, tokenizer, data recipe를 별도 기록한다.
5. official kernel 결과와 controlled probe 결과를 같은 표에서 혼동하지 않는다.

## KVEraser

1. Graph-COM official repository와 exact commit을 고정한다.
2. recurrent condition과 Transformer condition 모두 oracle localization을 받거나, 양쪽 모두 localization cost를 포함한다.
3. `INVALIDATE`는 KVEraser, `SUPERSEDE`는 erase-old + append-new로 비교한다.
4. full/suffix re-prefill을 exact reference로 둔다.
5. E07 mock readiness report는 scientific evidence가 아니다.
