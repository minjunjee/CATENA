# CATENA v6.1.0

**Constrained Behavioral Reachability for Transactional Control in Finite Memory**

CATENA v6.1 tests whether the target update demanded by an external transaction lies in the finite-memory controller's **bounded behavioral control set**, and whether the remaining reachability gap predicts correction-retention error.

## Active E00/E01 safety

Do not mutate the live repository while `e00_protocol_lock.py` or `e01_local_controllability.py` is active. Those runs remain v6.0 pilots. v6.1 adds the confirmatory H1 as:

```text
experiments/e01b_constrained_behavioral_reachability.py
```

Apply the post-E01 patch only after the two active processes terminate. See `docs/POST_E01_PATCH_INJECTION_KO.md`.

## Workshop critical path

| Stage | Question | Status |
|---|---|---|
| E01b / H1 | Does constrained behavioral regret predict unseen-geometry learned error? | Core |
| E02 / H2 | Does erase-write factorization help asymmetric demand without symmetric/retention damage? | Core |
| E03 / H3 | Is diagonal control sufficient exactly when demand operators are jointly diagonalizable? | Parallel theory |
| E04 / H4 | Is the factorization advantage functionally mediated by operation-matched gates? | Strong core |
| E05 / H5-lite | Does the direction survive held-out SUPERSEDE without operation labels? | Small anchor |
| E06 / H6 | Is one-time assimilation reusable and cost effective? | Post-workshop |
| E07 / RQ-T | What is the Transformer/KVEraser boundary? | Post-workshop |

## Verification

```bash
python -m pytest -q
python -m compileall -q src experiments tests
python tools/run_all_dry.py --device cpu --artifact-root artifacts_dry_run
```

Reference/mock outputs validate equations and orchestration only. They do not open scientific claims.
