# 4× RTX PRO 6000 Blackwell 서버 설정

## 1. 권장 환경

- Python 3.11
- PyTorch 2.12.1 + cu130
- NVIDIA driver 580.126.16
- System CUDA toolkit 13.0
- BF16

PyTorch wheel은 CUDA runtime을 포함한다. `/usr/local/cuda`는 FLA/RWKV custom extension을 컴파일할 때 사용한다.

```bash
bash scripts/00_bootstrap_and_audit.sh
source .venv/bin/activate
source scripts/setup_paths.sh
```

## 2. 실제 장치 확인

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
python -m catena.cli audit
```

GPU 제품명, VRAM, MIG, P2P는 이 출력으로 확정한다. 네 장을 한 DDP job으로 묶는 것이 기본값이 아니다.

## 3. 모델 cache와 extension build

```bash
export CATENA_SCRATCH=/fast-local-scratch/$USER/catena
source scripts/setup_paths.sh
bash scripts/install_rwkv_fla.sh
```

`HF_HOME`와 `TORCH_EXTENSIONS_DIR`은 NFS가 아닌 local SSD를 권장한다. FLA commit은 반드시 `FLA_REF=<commit>`으로 pin하고 `artifacts/logs/fla-commit.txt`에 남긴다.

## 4. 장시간 실행

```bash
bash scripts/run_in_tmux.sh catena-h1h2 "bash scripts/03_h1_h2_pilot_4gpu.sh"
tmux attach -t catena-h1h2
```

각 GPU lane의 stdout/stderr와 PID는 `artifacts/logs/<RUN_ID>/`에 저장된다.
