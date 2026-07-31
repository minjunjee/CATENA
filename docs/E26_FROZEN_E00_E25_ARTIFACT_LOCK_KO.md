# E26 이전 완료 evidence artifact 불변성 lock

## 목적

E26 admission은 기존 `POST_E21_PREIMPLEMENTATION_ARTIFACT_SHA256.json`의
E00–E21 1,329개 파일만 검사해서는 안 된다. 그 baseline 생성 뒤 완료된
E22–E25의 boundary·negative·audit-pending disposition도 모두 보존해야 한다.

`E26_FROZEN_E00_E25_ARTIFACT_LOCK.json`은 기존 baseline을 수정하지 않고
다음 두 층을 결합한다.

| 층 | 파일 수 | aggregate SHA-256 |
|---|---:|---|
| 기존 E00–E21 baseline | 1,329 | `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` |
| E22–E25 + `POST_E21_WAVE1_STATUS.json` | 733 | `3b9524854ee01d17a9a3f99b8b0ebd08a2ebf0c725d3765dac3496442772564e` |
| 완료 evidence 전체 | 2,062 | `46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b` |

E22–E25 extension은 다음 canonical namespace를 각각 file count, total bytes,
path/bytes/content-SHA aggregate로 고정한다.

```text
e22a_locality_method_selection
e22b_active_path_locality
e23a_product_poset_screen
e23b_product_poset_confirmatory
e24a_approximate_rank_stress
e24b_behavioral_attainability_stress
e25a_official_gdn2_gate
e25b_human_audit_packages
e25b_text_transaction_anchor
POST_E21_WAVE1_STATUS.json
```

E25a terminal failure와 E25b `AUDIT_PENDING`은 positive evidence로 승격하는
것이 아니라, 이미 확정된 disposition을 immutable scope에 포함하는 것이다.

## 제외 범위

Inventory parser는 lowercase experiment directory와 uppercase top-level
experiment JSON의 번호를 파싱하고 최대 번호를 25로 고정한다. 따라서 다음은
불변성 aggregate에 포함하지 않는다.

- E26 이상 canonical/new artifact
- `_launcher_logs`, `_launcher_locks`와 underscore-prefixed operational shard
- E26 non-evidence `/tmp` preflight

E26 artifact가 나중에 생성되어도 pre-E26 evidence aggregate는 변하지 않지만,
E22–E25 파일의 추가·삭제·변경 또는 새 E22–E25 namespace는 gate를 닫는다.

## 사용

`create_e26_frozen_invariance_receipt.py`의 `--baseline-manifest`에는 과거
E00–E21 baseline이 아니라 composite lock을 전달한다.

```bash
--baseline-manifest \
  /home/minjun_dev/CATENA_E26/docs/E26_FROZEN_E00_E25_ARTIFACT_LOCK.json
```

과거 baseline만 직접 전달하면 `FrozenInvarianceError`로 fail-closed한다.
