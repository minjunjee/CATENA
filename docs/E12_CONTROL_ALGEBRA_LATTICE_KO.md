# E12 - Architecture-Demand Control Lattice

## 목적

최근 recurrent-memory architecture가 추가하는 자유도를 같은 protocol에서 분리한다.

| Demand | 필요한 자유도 | 비교 |
|---|---|---|
| asymmetric same-address update | erase/write magnitude 분리 | tied → dual |
| partial value-subspace update | channel granularity | dual → diagonal |
| erase와 write의 다른 address | address decoupling | diagonal → separate-address |
| 같은 descriptor, state에 따라 다른 update | stored-content conditioning | separate-address → state-aware |

## 공정성

모든 variant는 동일한 maximal head를 등록하며, reachable output projection만 다르다. 같은 data order, optimizer, steps, state/value dimensions를 사용한다.

## Primary estimand

각 adjacent architecture pair가 정확히 대응하는 demand family에서 얻는 selective affected-MSE gain과, 더 단순한 family에서의 non-inferiority.

## Claim gate

4개 selective interaction이 모두 통과해야 control-lattice claim을 연다. 한 모델의 aggregate 평균 우위만으로는 통과하지 않는다.

## 의미

성공하면 GDN2, granularity, EDA, state-aware update를 단일 설계 지도로 연결하는 full-paper 중심 결과가 된다. 공식 architecture 성능 claim은 아니다.

## 2026-07-27 artifact-completion amendment

### 기존 supported 결과는 immutable

다음 run은 E12의 등록된 네 selective contrast를 모두 통과한 기존
`CONTROLLED_REFERENCE` 결과다.

| 항목 | 고정값 |
|---|---|
| Run ID | `20260727T182449.721061Z` |
| Execution status | `PASS` |
| Claim gate | `SUPPORTED` |
| Report SHA-256 | `5d300ea84fbe004370a2a44854637b2d60a0a0ccf5c63fc2ee2a24bce8fa2562` |
| Run-manifest SHA-256 | `f1545c0cbe8753bcc4b7e390a4586ec0bcd9f555b340f5c7e9beb248717dc1a0` |
| Metrics SHA-256 | `a2019187f0b6cebf45f1350964d8594357b259823e7a058e67912a583ae99058` |

이 run의 report, metrics, 판정은 수정하거나 재분류하지 않는다. 해당 run에는
학습 checkpoint와 schema-v2 provenance artifact가 없으므로, 결과는
**supported이지만 artifact-completeness가 불충분한 원본 run**으로 보존한다.

### amendment 범위

새 UTC run은 원본 결과를 덮어쓰거나 사후 재판정하기 위한 실험이 아니라,
동일 protocol을 다시 실행하여 artifact 계약을 완성하는 replication이다.

변경하지 않는 항목:

- `configs/e12_control_algebra_lattice.yaml`의 data, seeds, model grid, training,
  evaluation, statistics
- affected-MSE, retention-MSE와 네 adjacent selective contrast의 정의
- `selective_gain=0.001`
- `simpler_task_noninferiority=0.0005`
- `alpha=0.05`
- allowed/forbidden claim 문구와 `CONTROLLED_REFERENCE` evidence boundary

추가하는 항목:

- seed×controller별 checkpoint 40개(8 seeds × 5 controller freedoms)
- 각 metric row의 checkpoint 절대경로와 checkpoint SHA-256
- `config.resolved.yaml`
- `environment.json`
- schema-v2 `run_manifest.json`
- 명시적 `run_mode=MAIN`

기존 run의 metric row나 checkpoint를 새 run에 재사용하지 않는다. 새 run은
동일한 고정 config로 학습과 평가를 처음부터 수행한다. 새 artifact의 수치가
기존 결과와 달라질 경우에도 기존 report를 변경하지 않고, 두 run을 별도
record로 남긴다.

### 사전 lock

재실행에 사용되는 exact source/config hash와 위 원본 artifact hash는
`docs/E12_ARTIFACT_COMPLETION_AMENDMENT_LOCK.json`에 평가 시작 전에 고정한다.

### claim boundary

이 amendment는 artifact 및 재현성 계약만 보완한다. E12는 계속
controlled finite-memory reference evidence이며, official backend,
pretrained language model, runtime 우위 또는 보편적 architecture 우월성
claim에는 사용할 수 없다.
