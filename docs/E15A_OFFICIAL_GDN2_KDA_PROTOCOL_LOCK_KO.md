# E15a official GDN2/KDA protocol lock

동결 시각: `2026-07-28T06:12:00.901892Z`

## 목적

E15a는 reference recurrence가 아니라, 정확히 고정한 official GDN2/KDA
operator source가 parity·state·gradient·intervention 계약을 만족하는지
검사한다. 이 문서는 첫 `MAIN` 실행 전에 작성했으며, dry-run artifact는
scientific evidence에 포함하지 않는다.

## 고정 source와 runtime

| 항목 | 고정값 |
|---|---|
| GDN2 repository | `/home/minjun_dev/CATENA_official/gdn2_upstream` |
| GDN2 commit | `95709fc250357c2dd109361c353192f2aa5913f9` |
| Flash Linear Attention repository | `/data/minjun_dev/CATENA/official_sources/flash-linear-attention_19b5a3f_clean` |
| FLA commit | `19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5` |
| 독립 environment | `/data/minjun_dev/CATENA/envs/gdn2_official_95709fc` |
| Python | `3.11.15` |
| PyTorch / CUDA runtime | `2.9.0+cu128` / `12.8` |
| Triton | `3.5.0` |
| GPU family | NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0 |
| NVIDIA driver | `580.126.16` |

GDN2의 tracked `chunk_gdn2.py`와 `chunk_kda.py`를 직접 import한다.
GDN2 package initializer가 요구하는 unrelated FlashAttention stack은
operator gate의 dependency가 아니므로 실행하지 않는다. 이 격리는
reference/mock 구현으로 대체하는 것이 아니며, import된 모든 `lit_gpt`
및 `fla` Python source가 위 두 pinned checkout 아래의 tracked file인지
검증한다.

## 사전 고정 gate

| Gate | 판정 |
|---|---|
| Full sequence 대 carried chunks FP32 | relative L2 `<= 1e-5` |
| Tied GDN2 reduction 대 KDA FP32 | relative L2 `<= 1e-5` |
| BF16 대 FP32 | relative L2 `<= 5e-3` |
| Backward | 모든 등록 input gradient finite |
| State | carry·clone·restore 계약 통과 |
| Intervention | erase/write input hook confinement 및 causal-prefix 보존 |

Sequence length는 96, carry split은 64, batch/head는 각각 1,
key/value dimension은 각각 16, probe seed는 150015로 고정한다.

## Evidence 경계

여섯 gate와 pinned revision 검사를 모두 통과한 `MAIN`만
`official_operator_claim_eligible=true`가 될 수 있다. PASS하더라도
허용 범위는 official operator parity와 intervention hook 동작까지다.
E02b/E12/E13의 official replication, pretrained language model,
natural-language transaction, agent behavior 또는 systems superiority는
별도 실험 전에는 열리지 않는다.

`docs/E15A_OFFICIAL_GDN2_KDA_PROTOCOL_LOCK.json`은 source, config,
plugin과 runtime hash를 기계 판독 가능한 형태로 보존한다.
