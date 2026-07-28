# E25a Official GDN2/KDA Operator Gate Protocol

## 목적

E25a는 controlled reference backend 결과를 official GDN2 recurrence
operator에서 최소 복제하기 전에 실행하는 fail-closed gate다. Reference
또는 mock backend로의 fallback은 없다.

## 분리 환경

`catena-v6`은 변경하지 않는다. Gate는
`CATENA_E25A_ENV_PREFIX`와 실제 `sys.prefix`가 정확히 같은 별도 prefix에서만
실행된다. 재현 환경 명세는 다음 파일에 있다.

```text
environments/e25a_official_gdn2_environment.yaml
```

이 YAML은 **재생성용 명세**이며 설치 사실을 증명하지 않는다. 현재 별도
prefix를 변경하지 않고 읽어서 기록한 exact inventory는 다음 파일이다.

```text
environments/e25a_official_gdn2_observed_lock.json
```

관찰된 runtime은 Python 3.11.15, torch 2.9.0+cu128, Triton 3.5.0,
einops 0.8.2, ninja 1.13.0, NumPy 2.4.6, FLA distribution 0.5.2다.
Ambient editable FLA는 pinned revision이 아니므로 evidence source가 아니다.
Gate는 별도 pinned FLA checkout을 import path에 우선하고 실제 module
origin까지 검증한다.

현재 서버에서 read-only audit로 확인한 기존 후보:

```text
environment:
  /data/minjun_dev/CATENA/envs/gdn2_official_95709fc
GDN2:
  /home/minjun_dev/CATENA_official/gdn2_upstream
  95709fc250357c2dd109361c353192f2aa5913f9
FLA:
  /data/minjun_dev/CATENA/official_sources/flash-linear-attention_4b02d15_clean
  4b02d15d6a68700181b180235be62a9fb95d2a38
adapter:
  /home/minjun_dev/CATENA_official/plugins/catena_official_plugins/gdn2_gate.py
  e5643656de1a9ba164f78c4bdd46a66b971acedf0f84a8114a7cf3b38ba575a3
```

이 경로는 config 환경 변수로 명시해야 하며, 코드가 자동 탐색하거나
대체하지 않는다.

## Source gate

실행 전에 다음을 모두 검증한다.

1. exact detached Git revision
2. tracked-source clean status
3. expected upstream remote
4. GDN2와 FLA license path/SHA-256
5. adapter module origin과 source SHA-256
6. replication stage에서는 replication module origin과 source SHA-256
7. 별도 Python prefix

## Numerical/operator gate

기존 E15 contract와 threshold를 낮추지 않고 다음을 요구한다.

| Check | Gate |
|---|---:|
| FP32 full vs chunk recurrence | relative L2 `<=1e-5` |
| scalar-tied GDN2 vs KDA | relative L2 `<=1e-5` |
| BF16 vs FP32 | relative L2 `<=5e-3` |
| backward | 모든 gradient finite |
| state carry/clone/restore | exact registered contract |
| intervention hook | 지정 erase/write gate에만 confined |

일부 check만 통과하면 전체 status는 `FAIL`이며
`scientific_evidence=false`다. Dependency가 없으면 `NOT_CONFIGURED`다.

## Gate 후 최소 replication

`--stage replication`은 다음을 모두 요구한다.

- explicit E25a gate `report.json` path
- gate report의 official source/parity PASS
- `--allow-scientific-replication`
- repository 안에 hash-lock된 official replication plugin

Registered subset:

1. E02b ADD/INVALIDATE magnitude contrast
2. E18 magnitude sequence subset
3. E22 safe locality가 열렸을 때만 locality subset

Replication plugin은 official `chunk_gdn2`와 `chunk_kda`만 호출한다.
Reference recurrence나 mock fallback은 없다. E02b는 등록된 seed, dimension,
norm/angle grid와 metric을 유지하되 official recurrence에서 oracle-candidate
행렬을 직접 주입할 수 없으므로 orthogonal-address operator projection으로
한정한다. E18은 등록된 sequence generator, update/gap grid, affected/retention
metric을 그대로 쓰며 official 최소 subset의 표본 수(2 batches × 32)는
원본 E18 평가 표본 수와 구분해 기록한다.

E22 subset은 `CATENA_E25A_E22B_REPORT`로 explicit MAIN report를 제공하고,
그 report가 controlled safe-locality pattern과 모든 guardrail을 통과했을
때만 대상이 된다. 다만 E22의 selected learned locality objective와 route가
pinned official operator path에 실제 구현되지 않은 현재 상태에서는 단순
dual retention check로 대체하지 않는다. Safe report가 제공되면 replication
전체를 `BLOCKED_DEPENDENCY / NOT_IMPLEMENTED`로 닫고 E22 row를 만들지
않는다. 경로가 없을 때만 registered optional skip으로 E02b/E18 subset을
진행한다. 잘못된 report를 skip으로 바꾸지 않고 실패한다.

Safe report 검증은 frozen E22b schema의
`recovery_pattern`, `capacity`, `absolute_locality`, `retention`,
`locality_retention`, `selected_vs_mean_locality` gate를 모두 직접 확인하며,
8-seed MAIN run과 exact safe status를 요구한다.

Replication artifact의 raw JSONL은 wrapper 한 행이 아니라 plugin이 반환한
실제 subset×seed 행을 저장한다. 각 행에는 official source와 gate dependency
SHA-256 provenance가 추가된다.

Gate report를 latest glob으로 선택하지 않는다. Official replication은
사용자 승인 전 실행하지 않는다.

## 실행

Dry contract:

```bash
python experiments/e25a_official_gdn2_gate.py \
  --config configs/e25a_official_gdn2_gate.yaml \
  --device cpu \
  --artifact-root /tmp/catena_post_e21_dry \
  --dry-run
```

별도 환경 parity gate:

```bash
export CATENA_E25A_ENV_PREFIX=/data/minjun_dev/CATENA/envs/gdn2_official_95709fc
export CATENA_GDN2_REPO=/home/minjun_dev/CATENA_official/gdn2_upstream
export CATENA_FLA_REPO=/data/minjun_dev/CATENA/official_sources/flash-linear-attention_4b02d15_clean
export CATENA_E25A_PLUGIN_SOURCE=/home/minjun_dev/CATENA_official/plugins/catena_official_plugins/gdn2_gate.py
export PYTHONPATH=/home/minjun_dev/CATENA/src:/home/minjun_dev/CATENA:/home/minjun_dev/CATENA_official/plugins

"$CATENA_E25A_ENV_PREFIX/bin/python" \
  experiments/e25a_official_gdn2_gate.py \
  --config configs/e25a_official_gdn2_gate.yaml \
  --device cuda:0 \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

Gate PASS는 operator parity claim만 연다. GDN2/KDA 성능 우위는 최소
replication까지 통과한 뒤에만 별도로 열린다.
각 `RESULTS_SUMMARY_KO.md`는 45줄 이하이며 report에 line count를 기록한다.

Replication 실행 시에는 위 환경 변수에 더해 필요할 때만 다음을 명시한다.

```bash
export CATENA_E25A_E22B_REPORT=/explicit/e22b/main/run/report.json
```
