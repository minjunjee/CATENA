# CATENA E26 Stage-3D 결정 및 실행 계약

## 결정

Stage-3C의 다음 판정은 영구 보존한다.

```text
BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE
```

Stage-3C threshold, namespace, report, raw artifact를 수정하거나 재판정하지
않는다. Dense FP32 확대와 counterfactual-layout padding도 Stage-3C repair로
사용하지 않는다.

Fresh successor인 `E26_STAGE3D`는 E26의 실제 estimand에 필요한 한 가지
사전 고정 physical layout의 admissibility만 검사한다. Scientific E26a의 결과를
생성하지 않으며, Stage-3D가 통과한 경우에만 non-evidence resource preflight를
실행한다.

## 지시 provenance

사용자가 전달한 외부 packet SHA-256은 다음과 같다.

```text
5982f03d8331dd7ff1ee802f8a71e271f9345aae6f728c811ceb839faf37cc98
```

해당 packet bytes와 `CATENA_E26_STAGE3D_DECISION_AND_CODEX_TASK_KO.md` 원본은
서버의 `/home/minjun_dev`, `/data/minjun_dev`, `/tmp`에서 발견되지 않았다.
그러므로 이 SHA를 검증됐다고 기록하지 않는다. 이번 prospective contract의
operative source는 2026-08-02 사용자 메시지와 repository에 추가되는 YAML 및
protocol lock이다.

## 고정 판정 구조

- G0: Stage-3C와 E00-E25 불변성, fresh namespace
- G1: Projected-Tied/Dual parameter, initialization, token/data cursor 및 physical
  layout identity
- G2: Stage-3C FP32 결과의 exact-hash 재검증
- G3: fixed layout에서 BF16 compiled backend와 기존 `reference_python`, 그리고
  BF16/FP32 reference distance
- G4: 별도 process의 same-layout replay A/B
- G5: token normalization, clip, AdamW, scheduler 및 finite-gradient integrity
- G6: compiled backend, graph-break/fallback 및 variant-specific override 부재

`12/12`는 3 candidates × 2 variants × 2 state contexts를 뜻한다. 각 case는
두 G3 comparison을 모두 포함한다. Replay는 3 candidates × 2 variants의 6개
case다.

기존 `reference_python`을 사후에 다른 oracle로 바꾸지 않는다. Stage-3C에는
counterfactual layout sensitivity와 별개로 fixed monolithic
compiled-vs-reference BF16 mismatch가 이미 존재하며, Stage-3D artifact에 이를
별도 diagnostic으로 기록한다.

## Physical layout

Outcome을 보기 전에 다음을 고정한다.

| Candidate | Context | Microbatch sequences/GPU | Global input tokens | Accumulation |
|---|---:|---:|---:|---:|
| `d512_ctx4096` | 4096 | 1 | 65,536 | 16 |
| `d512_ctx2048` | 2048 | 1 | 65,536 | 32 |
| `d448_ctx4096` | 4096 | 1 | 65,536 | 16 |

두 variant는 후보 안에서 동일 layout, precision, data order, optimizer boundary와
backend를 사용한다. Variant-specific OOM fallback이나 layout selection은
허용하지 않는다.

## 종료 조건

```text
all G0-G6 pass:
  STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE

any hard gate fails:
  STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY

implementation/operational coverage incomplete:
  STAGE3D_NOT_EVALUABLE_IMPLEMENTATION_OR_EXECUTION_ERROR
```

GO일 때만 동일 fixed layout의 non-evidence resource preflight를 실행한다.
Resource projection은 model당 250M, 375M, 500M tokens, 168시간, 1.25 safety
multiplier, 100 GiB checkpoint cap을 사용한다. 어느 경우에도 Scientific E26a,
E26b 또는 E26c를 자동 시작하지 않는다.

Launcher는 Stage-3D process exit `0`뿐 아니라 canonical report의 GO disposition을
재확인한 뒤에만 resource runner를 호출한다. Resource runner는 Stage-3C에 잠긴
E26a config/tokenizer/corpus의 exact path+SHA를 다시 검증하고 canonical artifact
namespace에 receipt, worker binding, artifact audit와 latest pointer를 남긴다.
Resource exit `0/1/2`는 각각 feasible/infeasible/dependency-or-execution-error로
고정하며 어떤 exit도 Scientific E26a 승인으로 사용하지 않는다.

`NOT_EVALUABLE`은 새로운 과학적 결과가 아니라 incomplete execution 상태다.
따라서 G3/G4 raw coverage가 완결되지 않은 run을 numerical BLOCKED로 묶지 않는다.
