# E26 Final Result

## Disposition

```text
execution_status: BLOCKED_ADMISSION
scientific_disposition: BLOCKED_OFFICIAL_RUNTIME_NAMESPACE_PROVENANCE_VALIDATION
scientific_main_started: false
lm_transfer_claim_open: false
```

E26 Final did not test its scientific hypothesis. The official source pin,
community-checkpoint transport and strict structural compatibility, tokenizer
bytes, and the CUDA 13 / PyTorch 2.9 / FlashAttention / API-compatible FLA
dependency provenance passed. The canonical official-runtime admission did not.

The exact source pin is NVLabs `GatedDeltaNet-2` commit
`95709fc250357c2dd109361c353192f2aa5913f9`, licensed under the non-commercial
`NVIDIA Source Code License-NC`. The starting community checkpoint is
`model-100b.pth`, 17,401,727,659 bytes, SHA-256
`0322ebeefa96badb24d6b4b511c36b02374b704dc1a65b90eab2ee1383a9ce23`.
It is not an official NVIDIA weight and is byte-identical to the repository's
`model-95b.pth`, so the 100B training-token identity is not established. The
TinyLlama tokenizer revision is
`ff3c701f2424c7625fdefb9dd470f45ef18b02d6`, but it is not cryptographically
linked to the community checkpoint. The source-license/model-card-license
difference remains a manual-review boundary.

The first Dual and Projected-Tied attempts stopped before GPU execution because
implementation validation had left Python bytecode caches in the derived
checkout. Both blocked receipts were retained. Under the prospective R1
amendment, only those caches were removed and the unchanged audit was run once
in fresh namespaces. Both R1 variants then stopped with:

```text
E26FinalCheckpointAuditError:
Refusing preloaded non-official Python module: lit_gpt.gdn2_ops
```

The imported object appears to be the pinned checkout's namespace package,
which does not satisfy the audit's ordinary-module `__file__` provenance check.
The registered R1 was already consumed, so the checker was not repaired and no
additional attempt was made.

No canonical 1.3B kernel audit, speed preflight, bridge, main training, frozen
evaluation, mechanism intervention, quality guardrail, or systems benchmark was
run. Consequently this is neither a null result nor a refutation of E26; it is
an official-runtime admission failure. No pretrained-LM, official-operator,
transaction-effect, mechanism, quality/locality, or speed claim is eligible.
Data, throughput, VRAM, power, bridge quality, and all scientific metrics are
therefore `NOT RUN` or `NOT MEASURED`. The dependency receipt passed provenance
only: external decode-cache clone/restore plumbing remained unimplemented and
`decode_cache_evaluation_eligible=false`. This independent later-stage
limitation did not change the earlier namespace-blocker execution branch.

The Stage-3C/3D dispositions remain unchanged, and the frozen E00--E25 aggregate
(2,062 files; SHA-256
`46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b`)
was revalidated.

Canonical terminal artifact:
`/data/minjun_dev/CATENA/artifacts/e26_final_gdn2_1p3b_transactional_transfer/20260803T161043.290986Z`
