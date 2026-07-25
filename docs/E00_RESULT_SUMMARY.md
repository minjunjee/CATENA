# E00 환경 감사 결과 요약

기준 산출물은 `artifacts/profiles/e00_audit/latest.json`과 그 파일이 가리키는
run directory다. 현재 판정은 **PASS**이며 필수 check의 실패나 차단은
없다. E00은 실행할 때마다 결과 원본, 설정 snapshot, 소스 fingerprint,
패키지·Conda snapshot과 SHA-256 manifest를 새 run directory에 보존한다.

| 영역 | 결과 | 관측 및 해석 |
|---|---:|---|
| 실행 환경 | 통과 | 실제 `catena` prefix의 Python 3.11.15와 glibc 2.39를 확인했다. `torch==2.12.1+cu130`, CUDA runtime 13.0과 base/model/train/dev 의존성이 선언 범위를 만족하며 `pip check`도 통과했다. |
| GPU/driver | 통과 | 호스트는 계획의 4장이 아니라 동일한 RTX PRO 6000 Blackwell Server Edition 8장을 노출한다. 표준 실험 할당은 물리 GPU 0–3 네 장으로 고정했다. 각 장은 97,887 MiB, compute capability 12.0, MIG Disabled이며 driver는 580.126.16이다. topology와 P2P read/write도 원본 감사 자료로 보존했다. |
| CUDA/BF16 | 통과 | `nvcc` 13.0으로 실제 `sm_120` 커널을 빌드·실행했다. GPU 0–3의 독립 동시 프로세스가 각각 한 장만 보며 UUID/PCI가 inventory와 일치했다. 네이티브 BF16-input cuBLAS GEMM과 PyTorch BF16 matmul은 네 lane 모두 finite, max absolute error 0이었다. 대략 5 ms와 50–100 ms인 작은 기능 probe이므로 처리량 결론에는 쓰지 않는다. |
| 저장장치 | 무결성 통과·용량 경고 | state/model cache 각각에서 64 MiB 쓰기·`fsync`·읽기·SHA-256 검증을 3회 통과했다. 중앙값은 쓰기 약 0.45 GiB/s, 읽기 약 0.52 GiB/s였으나 읽기는 page cache 영향을 받을 수 있다. 가용 공간 약 25 GiB는 권장 128 GiB보다 작다. |
| 저장소 | 통과 | Python compile, 26개 shell syntax, 33개 pytest, 32개 config audit와 Mock end-to-end smoke를 통과했다. 두 미참조 eval config는 비차단 경고다. config snapshot, source-tree fingerprint, package/Conda 목록, raw 결과와 자체 검증되는 SHA-256 manifest를 보존했다. dirty tree는 hash로 식별한 구현 작업분이므로 경고로 기록했다. |

## 해석과 계획 영향

E00은 인프라 gate이므로 이 결과는 H1–H4를 지지하거나 반박하지 않는다.
**과학적 가설·metric·ablation·gate 순서의 변경은 없다.** E01을 시작할
수 있는 인프라 선행조건은 열렸다. 다만 실행 계획에는 다음 운영 변동이
있다.

1. 기본 4-lane 실험은 같은 NUMA 권역의 물리 GPU 0–3에 고정하고, GPU 4–7은
   별도 계획 변경 없이는 결과 수나 seed 수를 늘리는 데 사용하지 않는다.
2. 모델 다운로드, exact-teacher cache, checkpoint 생성 전에는 저장소
   가용 공간을 권장 128 GiB 수준으로 확보하거나 저장소 내부의 다른
   승인된 위치를 정해야 한다.
3. FLA/RWKV backend 설치처럼 `catena` 패키지 snapshot을 바꾸는 작업 뒤에는
   E01보다 먼저 E00을 다시 PASS해야 한다.
4. 기존 7월 24–25일 E01–E03 일정은 이미 지났으므로, 저장공간 확보와 E01
   완료 시점을 기준으로 달력만 다시 잡는다. 과학적 선후관계는 유지한다.
