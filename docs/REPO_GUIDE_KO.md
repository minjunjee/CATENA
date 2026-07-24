# 저장소 구조: 실험 단계에서 찾는 법

이 저장소는 모델 이름이 아니라 **실험이 진행되는 순서**를 기준으로 찾을 수 있게 구성되어 있다.

```text
CATENA_REALM2026_baseline_repo_v0.6.0/
├── configs/
│   ├── data/                 # E02: smoke/pilot/main/stress/chain split
│   ├── experiments/          # E00-E11: 한 파일이 한 실행 단위
│   └── models/               # RWKV inference/train 후보, Qwen
├── data/
│   ├── raw/                  # 외부 원본이 생길 경우
│   ├── processed/            # generator가 만든 fixed JSONL splits
│   └── manifests/            # split hash와 generator version
├── src/catena/
│   ├── data/                 # E02 episode/chain generator와 validator
│   ├── methods/              # H2 text policies와 H3 encoder input
│   ├── models/               # recurrent state/KV cache 공통 adapter
│   ├── training/             # E05 teacher, E06-E07 H3, E09 H4
│   ├── experiments/          # inference eval, H3/H4 eval, profile, tool-call
│   ├── eval/                 # coherence aggregation, bootstrap, shard merge
│   └── utils/                # manifest, seed, timing
├── artifacts/
│   ├── logs/                 # E00 audit, process logs, pip freeze
│   ├── teacher_cache/        # E05/E08/E09 exact candidate distributions
│   ├── checkpoints/          # H3/H4/Transformer soft patch
│   ├── metrics/              # raw predictions와 summary
│   ├── profiles/             # E10 system measurements
│   └── figures/              # 논문용 최종 figure
├── scripts/
│   ├── 00_...12_...sh        # 시간 순서의 4-GPU entry points
│   ├── launch_4gpu.sh        # 네 독립 CUDA_VISIBLE_DEVICES lane
│   ├── run_in_tmux.sh        # SSH와 분리한 persistent 실행
│   └── audit/install/...     # 환경과 backend 준비
├── docs/                     # 실험 명세, claim gate, backend 상태
└── tests/                    # CPU/mock/unit tests
```

## 설정 파일과 산출물의 연결

예를 들어 H3 main 학습은 다음 경로를 따른다.

```text
configs/data/main.yaml
        ↓ E02
 data/processed/main/{train,val,test}.jsonl
        ↓ E05 exact teacher
 artifacts/teacher_cache/main/*_teacher_scores.jsonl
        ↓ E06-E07
 configs/experiments/e06_h3_slots{4,8,16}.yaml  # E06 sweep
        ↓ validation selection
 configs/experiments/e07_h3_main.yaml        # E07 3 seeds
        ↓
 artifacts/checkpoints/e07_h3_main/seed_*/encoder_final.pt
        ↓ E07
 artifacts/metrics/e07_h3/.../{catena_predictions.jsonl,summary.json}
```

Transformer는 같은 data를 사용하지만 별도의 exact teacher와 checkpoint namespace를 쓴다.

```text
data/processed/main
  ├─ artifacts/teacher_cache/qwen_main
  ├─ artifacts/checkpoints/qwen_soft_patch
  └─ artifacts/metrics/e08_transformer
```

## Run directory 원칙

각 run은 최소 다음을 보관한다.

- 실행에 사용한 resolved config
- git commit SHA와 dirty 상태
- host audit path/hash
- model ID/checkpoint hash
- seed와 data manifest hash
- raw prediction/logits
- aggregated metric
- latency samples

논문 표는 raw prediction으로부터 재생성하고, summary JSON만 수기로 옮기지 않는다.

## 새 실험을 추가하는 방법

1. `configs/experiments/eXX_name.yaml`을 만든다.
2. 기존 공통 model/data adapter를 재사용한다.
3. 출력 경로를 `artifacts/<type>/eXX_name`으로 둔다.
4. CLI subcommand가 없으면 `src/catena/experiments/`에 runner를 추가한다.
5. toy/mock unit test와 config smoke를 추가한다.
6. `EXPERIMENT_RUNBOOK_KO.md`와 `CLAIM_GATES_KO.md`에서 어떤 claim을 검증하는지 연결한다.
