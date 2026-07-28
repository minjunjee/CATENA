# 실행 중인 E00/E01 뒤에 v6.1을 joint injection하는 방법

## 원칙

현재 `e00_protocol_lock.py` 또는 `e01_local_controllability.py`가 실행 중이면 live repository를 수정하지 않는다. 두 run은 v6.0 fingerprint로 끝까지 보존하고, 결과는 pilot/construct diagnostic으로만 사용한다. v6.1의 confirmatory H1은 새 entry point `e01b_constrained_behavioral_reachability.py`에서 시작한다.

## 절대 덮어쓰지 않는 frozen dependency closure

```text
experiments/e00_protocol_lock.py
experiments/e01_local_controllability.py
experiments/common.py
configs/e00_protocol_lock.yaml
configs/e01_local_controllability.yaml
src/catena/core/io.py
src/catena/core/randomness.py
src/catena/core/schema.py
src/catena/data/tamp.py
src/catena/eval/metrics.py
src/catena/eval/statistics.py
src/catena/models/controllers.py
src/catena/models/memory.py
src/catena/theory/control_geometry.py
src/catena/training/losses.py
src/catena/training/probe.py
```

Patch payload에는 위 파일이 포함되지 않는다. Apply script는 적용 전후 SHA-256을 비교해 한 바이트라도 달라지면 rollback한다.

## 적용 시점

```bash
pgrep -af 'e00_protocol_lock.py|e01_local_controllability.py'
```

아무 출력이 없을 때만 적용한다.

## 자동 적용

```bash
cd /home/minjun_dev
unzip CATENA_v6.1_post_E01_joint_patch.zip -d CATENA_v6.1_patch
PATCH_DIR=/home/minjun_dev/CATENA_v6.1_patch/CATENA_v6.1_post_E01_joint_patch

bash "$PATCH_DIR/APPLY_POST_E01_PATCH.sh" /home/minjun_dev/CATENA
```

Apply script는 다음을 수행한다.

1. active E00/E01 process 재확인
2. v6.0 frozen file hash 검증
3. 교체 대상의 timestamp backup
4. additive/replace payload 주입
5. frozen hash 재검증
6. compileall, pytest, E00-E08 CPU dry-run
7. source manifest와 patch receipt 생성
8. 실패 시 변경 대상만 자동 rollback

## 적용 후 실행 순서

```bash
cd /home/minjun_dev/CATENA

CUDA_VISIBLE_DEVICES=0 python -m experiments.e01b_constrained_behavioral_reachability \
  --config configs/e01b_constrained_behavioral_reachability.yaml --device cuda:0

CUDA_VISIBLE_DEVICES=0 python -m experiments.e02_magnitude_factorization \
  --config configs/e02_magnitude_factorization.yaml --device cuda:0

python -m experiments.e03_granularity_orientation \
  --config configs/e03_granularity_orientation.yaml --device cpu

CUDA_VISIBLE_DEVICES=1 python -m experiments.e04_functional_mediation \
  --config configs/e04_functional_mediation.yaml --device cuda:0

CUDA_VISIBLE_DEVICES=2 python -m experiments.e05_semantic_demand_inference \
  --config configs/e05_semantic_demand_inference.yaml --device cuda:0
```

E04는 E02의 8 seed checkpoint가 immutable해진 뒤에만 실행한다. E08 claim freeze 전에는 H5 audit CSV를 자동으로 채우지 않는다.
