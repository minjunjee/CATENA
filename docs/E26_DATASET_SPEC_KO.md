# E26 Dataset Implementation Specification

## 1. Canonical episode object

필수 metadata:

```text
episode_id
split
domain
operation
template_family
entity_namespace
seed
state_before
state_after
materialization_text
transaction_text
dependency_closure
distractor_text
branch_prefix_text
queries[4]
exact_refresh_text
protected_fields
```

Model-visible serialization과 evaluator-only metadata를 분리한다.

## 2. Query object

```json
{
  "query_type": "derived_action",
  "prompt": "...",
  "candidate_answers": ["...", "..."],
  "gold_index": 0,
  "structured_gold": {"tool":"..."},
  "affected_entities": ["..."],
  "retained_entities": ["..."]
}
```

## 3. Branch construction

```python
runtime = model.prefill(branch_prefix_tokens)
for query in queries:
    branch_state = runtime.clone(deep=True)
    score(query, state=branch_state)
```

Reference dry-run에서 re-prefill fallback을 쓸 수 있으나 manifest에 `branch_mode=REPREFILL_NON_EVIDENCE`를 기록한다. Scientific evaluation은 runtime clone을 요구한다.

## 4. Distractor generation

Distractor는 일반 corpus text와 structured entity events를 혼합한다. Target answer를 포함하지 않으며, entity/value collision table을 검사한다. Exact token gap은 tokenizer materialization 후 pad/truncate하여 맞춘다.

## 5. Near-duplicate audit

- canonical protected-field signature exact overlap
- normalized text SHA
- 5-gram MinHash/Jaccard threshold
- template AST identity
- entity/value pair overlap

Main test leakage가 하나라도 있으면 benchmark manifest를 새 namespace로 재생성하고 기존 failed audit를 보존한다.

## 6. Suggested sizes

E26a tiny validation:

- 100 items/operation
- all query types

E26b calibration:

- train stream generated online
- validation 2,000 episodes
- no main test access

E26d frozen benchmark candidate:

- 4 operations × 3 update levels × 4 gaps × 4 query types × 25 episodes = 4,800 branch evaluations per domain condition
- in-domain balanced total 9,600–14,400 episodes depending domain stratification
- held-out domain separate 2,400–4,800

Final count는 compute보다 seed-level uncertainty와 cell coverage를 기준으로 E26b 전에 lock한다.
