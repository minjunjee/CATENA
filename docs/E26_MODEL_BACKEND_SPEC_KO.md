# E26 Optimized Backend Specification

## 1. Reference vs scientific path

`recurrent_mixer.py`의 token loop는 equation correctness를 위한 oracle reference다. MAIN backend는 reference와 같은 equations와 state layout을 사용하되 token dimension을 Python에서 순회하지 않는다.

## 2. 허용 구현

### Option A — compiled fixed-size chunks

- chunk size 32/64/128
- inner scan captured by `torch.compile(fullgraph=True)` 또는 higher-order scan
- outer Python loop는 chunk 수만 순회 가능
- state carry tensor shape fixed

### Option B — Triton/CUDA kernel

- pinned source commit
- forward/backward implementation
- BF16 accumulation policy documented

### Option C — existing FLA primitive adaptation

- official upstream equation과 다르면 reference adapter로 명확히 구분
- upstream commit/license/module origin 기록
- no silent fallback

## 3. Required parity grid

- batch 1/2/4
- seq 1/31/32/33/127/128/129/512
- random initial state and zero state
- continuation split points 1/17/64/T-1
- Dual and Projected-Tied
- FP32 full/chunk relative L2 ≤1e-5, max abs ≤1e-5
- BF16 vs FP32 relative L2 candidate ≤7e-3; final threshold E26a lock
- backward finite and relative gradient check on tiny tensors

## 4. Compile audit

- graph break count
- recompilation count across batch/seq candidates
- compile wall time
- generated kernel cache path/hash
- fallback counter

Fallback during MAIN >0 is failure unless explicitly registered for unsupported tail length and numerically tested.

## 5. Throughput measurement

- 20 warm-up, 100 measured steps minimum
- CUDA synchronize around timing
- report mean/p50/p95
- actual non-padding tokens/s
- checkpoint I/O separate
- peak memory reset before measurement

## 6. Intervention hooks

E26e hook가 optimized path에도 작동해야 한다. Hook를 켰을 때 non-transaction token gates/state가 unchanged인지 parity test한다.
