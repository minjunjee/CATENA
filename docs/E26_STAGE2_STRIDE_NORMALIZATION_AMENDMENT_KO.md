# E26 Stage-2 compiled-scan stride normalization amendment

## 범위

이 amendment는 scientific E26a 이전의 `NON_EVIDENCE_VALIDATION`에서 발견된
implementation defect만 기록한다. 실험 threshold, seed, metric, model equation,
variant projection 및 data는 변경하지 않았다.

## 최초 실패

- 기준 Git HEAD: `5ea0233cd7b2a38b7292b016203df65ad8618129`
- 수정 전 `src/catena/lm/recurrent_mixer.py` SHA-256:
  `445723f63370a5973a37c5536ddfe17c39b0ede3f2b97b4b9665f65166762eed`
- 명령:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
pytest -q -m e26_gpu tests/test_e26_arbitrary_chunk_partitions.py
```

- 판정: `FAIL` (scientific evidence 아님)
- 핵심 trace:

```text
torch._dynamo hit config.recompile_limit (8)
function: _fixed_chunk_scan_...
tensor 'q' stride mismatch at index 0.
expected 256, actual 696
```

원인은 외부 arbitrary partition마다 projection tensor의 parent sequence stride가
달라졌지만, 동일 fixed-shape compiled scan code object가 stride별로 재compile된
것이다. 수치 parity를 평가하기 전에 compile cache limit에서 종료됐다.

## Prospective implementation repair

Compiled scan 진입 직전에 `q`, erase/write key, value, erase/write 및 decay의
fixed-chunk slice를 contiguous-format clone으로 정규화한다. 이는 tensor 값이나
recurrence equation을 바꾸지 않고 compiled boundary의 memory-layout signature만
고정한다. Dual과 Projected-Tied에 동일한 공용 mixer path로 적용된다.

최초 시도에서는 `.contiguous()`를 사용했으나 batch dimension이 1일 때 PyTorch가
해당 singleton stride를 보존해 동일 trace가 재발했다. 이 실패도 지우지 않았고,
명시적 `clone(memory_format=torch.contiguous_format)`으로 모든 stride를
canonicalize했다.

- 수정 후 source SHA-256:
  `4e3b9fffd47e3f948acbbb82d5e81b7d5b0b72ccc73dd53e87cd100dc546c756`
- Cache key에 stride를 추가하지 않음
- Existing FP32/BF16 tolerance 불변
- Mandatory partition 및 8 fixed-random partition 불변
- Monolithic reference와 output/state/KV/metadata/gradient comparison 불변

수정 후 검증은 최초 실패 namespace를 덮어쓰지 않는 fresh non-evidence run으로
수행한다. Canonical scientific E26a artifact는 이 amendment에서 생성하지 않는다.
