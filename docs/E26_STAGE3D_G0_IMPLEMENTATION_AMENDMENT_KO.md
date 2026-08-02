# E26 Stage-3D G0 구현 amendment

## 판정

최초 Stage-3D admission 시도는 다음 input-lock만 생성하고 G0에서 종료됐다.

```text
/data/minjun_dev/CATENA/e26_stage3d_input_locks/20260802T142626.974750Z
```

GPU worker, G3/G4 numerical case, resource preflight 및 Scientific E26a는
시작되지 않았다. 이 실패 input-lock은 삭제하거나 재판정하지 않는다.

## 구현 결함

Stage-3C가 생성한 historical frozen-invariance receipt의
`live_repository.observed_head`는 receipt 생성 당시 commit이다. 기존 G0는
현재 additive descendant에서 receipt를 다시 만든 뒤 historical JSON 전체와
동일해야 한다고 요구했다. 따라서 frozen base source와 E00--E25 artifact가
모두 정확히 보존됐어도 현재 HEAD가 달라진다는 이유만으로 admission이
실패했다.

이는 scientific/numerical failure가 아니라 dynamic observation을 immutable
contract로 잘못 취급한 implementation defect다.

## Prospective repair

fresh Stage-3D namespace에서 G0를 다음 순서로 수행한다.

1. Stage-3C protocol에 등록된 frozen receipt와 data-lock의 exact path 및
   byte SHA-256을 확인한다.
2. Historical receipt 내부 canonical SHA-256과 PASS contract를 확인한다.
3. 현재 clean descendant에서 source와 artifact structured audit을 새로
   계산한다.
4. Historical receipt와 fresh receipt에서 오직 다음 dynamic field만 비교에서
   제외한다.

   - `live_repository.observed_head`
   - `live_repository.head_matches`
   - top-level `receipt_sha256`

5. 그 외 모든 field는 exact match를 요구한다. 특히 ancestor 관계, clean
   status, frozen base-blob aggregate/count, missing/changed 목록, E00--E25
   artifact 2,062개와 aggregate/hash/namespace 목록은 완전히 동일해야 한다.

따라서 tracked base mutation, dirty worktree, frozen artifact drift, historical
receipt tampering은 계속 hard-block된다. Threshold, seed, layout, metric,
Stage-3C disposition 및 artifact는 변경하지 않는다.

## 실행 경계

이 amendment 구현과 test에는 corpus/main-test outcome 및 GPU를 사용하지
않는다. 수정 후 재실행은 기존 실패 input-lock을 재사용하지 않고 새 UTC
input-lock과 새 Stage-3D run namespace만 사용한다. Scientific E26a는 계속
닫혀 있다.
