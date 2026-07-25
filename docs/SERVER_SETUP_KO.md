# 4× RTX PRO 6000 Blackwell 서버 설정

## 1. 기존 실행 환경

- Python 3.11
- PyTorch 2.12.1 + cu130
- NVIDIA driver 580.126.16
- System CUDA toolkit 13.0
- BF16

PyTorch wheel은 CUDA runtime을 포함한다. `/usr/local/cuda`는 FLA/RWKV custom extension을 컴파일할 때 사용한다.

```bash
bash scripts/00_bootstrap_and_audit.sh
```

E00과 이후 stage script는 기존 Conda 환경 `catena`를 사용한다. E00은
패키지나 Conda 환경을 설치·수정하지 않으며 현재 환경을 hard gate로
검증한다.

## 2. 실제 장치 확인

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
PYTHONPATH=src conda run --no-capture-output -n catena python -m catena.cli audit
```

GPU 제품명, VRAM, MIG, P2P는 이 출력으로 확정한다. 네 장을 한 DDP job으로 묶는 것이 기본값이 아니다.

## 3. 모델 cache와 extension build

```bash
export CATENA_SCRATCH="$PWD/.scratch"
source scripts/setup_paths.sh
bash scripts/install_rwkv_fla.sh
```

모든 writable cache와 extension 출력은 저장소 내부에 둔다. FLA backend
설치를 포함한 `catena` 내부 package 작업은 승인돼 있다. FLA commit은
반드시 `FLA_REF=<commit>`으로 pin하고 `artifacts/logs/fla-commit.txt`에
남긴다. 다른 Conda 환경이나 system package는 변경하지 않는다.

실행할 때는 범위를 확인하는 보호 flag도 함께 지정한다.

```bash
CATENA_ALLOW_ENV_MODIFICATION=1 FLA_REF=<pinned-commit> \
  bash scripts/install_rwkv_fla.sh
bash scripts/00_bootstrap_and_audit.sh
```

환경 변경 뒤에는 반드시 E00을 다시 통과해야 다음 stage script의
package/source snapshot 검증이 열린다.

## 4. 장시간 실행

```bash
bash scripts/run_in_tmux.sh catena-h1h2 "bash scripts/03_h1_h2_pilot_4gpu.sh"
tmux attach -t catena-h1h2
```

각 GPU lane의 stdout/stderr와 PID는 `artifacts/logs/<RUN_ID>/`에 저장된다.
