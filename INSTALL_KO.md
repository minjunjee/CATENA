# Clean Install

이 배포본은 기존 `/home/minjun_dev/CATENA`의 변경사항을 보존하거나 병합하지 않습니다.

## 1. 기존 repository 삭제 후 새 배포본 해제

ZIP 내부 최상위 폴더는 `CATENA/`입니다.

```bash
cd /home/minjun_dev
rm -rf CATENA
unzip /path/to/CATENA_control_geometry_v6.1.0.zip
cd CATENA
```

## 2. 새 Conda 환경 생성

```bash
/home/minjun_dev/miniconda3/bin/conda create -n catena-v6 python=3.11 -y
source /home/minjun_dev/miniconda3/bin/activate catena-v6
```

## 3. CUDA 13.0 PyTorch 설치

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

CATENA package와 개발 도구를 설치합니다.

```bash
python -m pip install -e '.[dev]'
```

## 4. 환경 경로 설정

```bash
cp .env.example .env
vim .env
```

대용량 저장공간이 있다면 `CATENA_ARTIFACT_ROOT`, Hugging Face cache, 공식 backend checkout 경로를 그 위치로 지정합니다.

```bash
set -a
source .env
set +a
```

## 5. 설치 검증

```bash
python -V
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

python -m pytest -q
python -m compileall -q src experiments tests
```

## 6. E00 실행

```bash
python -m experiments.e00_protocol_lock \
  --config configs/e00_protocol_lock.yaml \
  --device auto
```

Clean install에서는 E00과 E01 pilot 이후 E01b부터 confirmatory path를 실행합니다. 이미 E00/E01이 동작 중인 live repo에는 clean install을 하지 말고 post-E01 patch를 사용합니다. 각 실행은 `artifacts/<experiment_id>/<run_id>/`에 raw episode output, report와 checkpoint를 기록합니다.
