# E26 Final repository/source 사전 감사

상태: `ADMISSION_IN_PROGRESS`  
작성 시점: 2026-08-03 UTC

## 보존 경계

- frozen repository: `/home/minjun_dev/CATENA`
- frozen HEAD: `ce128ea6dd264f4b073a061729d0b23274ecb12e`
- Stage-3D scientific source commit: `47cbc68636367e32832c66ea57d1a827282ef447`
- Stage-3D disposition: `STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY`
- Stage-3D canonical report SHA-256:
  `4c4528bf35052423896b29dbc12944e9ad5df3ec2f87410a9688417297a42650`
- E26 Final branch: `exp/e26-final-official-gdn2-main`
- E26 Final worktree: `/home/minjun_dev/CATENA_E26_FINAL`

E26 Final은 Stage-3C/3D repair가 아닌 fresh prospective experiment다. 기존
source, protocol, threshold, report와 artifact를 수정하거나 재판정하지 않는다.

## 지시 source

사용자가 설명한 41-file E26 Final packet은 서버에서 발견되지 않았다. 따라서
대화에 제공된 `CODEX MASTER TASK`, scientific protocol 및 prospective YAML을
canonical 신규 지시로 캡처했다. 누락 packet의 mock/helper를 추측하여 복사하지
않고 live repository convention에 additive하게 구현한다.

## local source diff 분류

Frozen HEAD `ce128ea6`은 Stage-3D scientific source commit `47cbc686`의
descendant다. 두 commit 사이 변경은 Stage-3D terminal result 문서 추가로
분류한다. E26 Final 시작 전에 machine-readable blob diff와 category receipt를
별도 artifact에 기록한다.

## official source

- repository: `https://github.com/NVlabs/GatedDeltaNet-2`
- detached commit: `95709fc250357c2dd109361c353192f2aa5913f9`
- commit tree: `bec1976e3b1ab0fab519f60c73e36a3c0092da47`
- local path: `/data/minjun_dev/CATENA/external/gdn2_official`
- license: NVIDIA Source Code License-NC (research/evaluation only)

Checkout은 clean detached state다. Official code에서 `gdn2_1.3B` config와
`chunk_gdn2`/`fused_recurrent_gdn2`, 별도 `b_proj`/`w_proj`가 확인됐다.
Official kernel source는 vendor tree 안에서 수정하지 않는다.

## checkpoint provenance warning

지정된 `model-100b.pth` transport metadata는 revision, byte size 및 LFS SHA와
일치한다. 그러나 동일 repository의 `model-95b.pth`가 같은 blob/SHA를 가리킨다.
따라서 `100B-token checkpoint` 표기는 community uploader metadata이며 독립적으로
입증된 NVIDIA training receipt가 아니다. 이 경고는 final claim/report에 남긴다.
실제 admission은 지정 bytes의 SHA, `weights_only=True`, exact key/shape,
`strict=True`, finite tensors, parameter count, finite forward 및 official kernel
dispatch로 fail-closed한다.

## tokenizer boundary

TinyLlama tokenizer는 revision
`ff3c701f2424c7625fdefb9dd470f45ef18b02d6`로 잠근다. 이는 vocab 32,000,
BOS=1, EOS=2, UNK=0이다. Existing CATENA 16K tokenizer/memmap은 재사용하지
않는다. Upstream training code의 pad 처리와 맞춰 EOS-as-pad를 명시적으로
고정하되, checkpoint embedding/output shape 및 fixed-batch forward가 통과해야
한다. 이 revision이 community checkpoint pretraining에 실제 사용됐다는
cryptographic linkage는 없으므로 provenance limitation으로 기록한다.

## resource audit

- GPU: 8× RTX PRO 6000 Blackwell Server Edition, 각 약 95 GiB; protocol은 0–3만 사용
- `/data` 여유: 약 1.2 TiB
- RAM 여유: 약 1.4 TiB
- root filesystem 여유: 약 85 GiB — large artifacts 저장 금지

Raw hardware는 admission 가능한 규모다. Software import, official kernel dispatch,
1.3B strict load, 32K data, throughput 및 wall-clock budget은 아직 gate 대상이다.

## 현재 허용된 다음 작업

1. 지정 checkpoint/tokenizer download 및 immutable byte receipt
2. isolated `weights_only=True` inspection과 strict compatibility audit
3. official runtime import/kernel dispatch smoke
4. 32K natural-language data protocol 및 sealed manifest 생성
5. admission PASS일 때만 speed preflight와 bridge/main 진행

어느 hard gate라도 실패하면 다른 checkpoint, backend, precision, model 또는
threshold로 대체하지 않고 새 `BLOCKED_*` report를 작성한다.
