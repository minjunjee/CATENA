# Codex task order after v6.1 patch

1. Verify `python -m pytest -q` and compileall.
2. Run E01b dry-run, then the measured pilot.
3. Inspect `episode_geometry_metrics.jsonl`; do not proceed if the predictor is only operation identity.
4. Run one E02 paired seed and verify all raw metrics and no-op/oracle denominators.
5. Launch all 8 paired E02 seeds in one canonical single-GPU run. Do not split
   lanes until a hash-checked shard merge/aggregation command exists.
6. Run E04 only after E02 checkpoints and raw JSONL are immutable.
7. Run E03 in parallel because it does not depend on E02.
8. Run E05 only as a small external-validity anchor and export the audit sheet.
9. Fill the audit independently; do not auto-fill reviewer columns.
10. Execute E08 and use its allowed claims verbatim in the paper draft.

Do not implement or launch official KVEraser/GDN2 language-model experiments on the REALM critical path.
