# Repository 구조 v6.1

```text
CATENA/
├── experiments/
│   ├── e00_protocol_lock.py                 # running pilot, unchanged
│   ├── e01_local_controllability.py         # running pilot, unchanged
│   ├── e01b_constrained_behavioral_reachability.py
│   ├── e02_magnitude_factorization.py
│   ├── e03_granularity_orientation.py
│   ├── e04_functional_mediation.py
│   ├── e05_semantic_demand_inference.py
│   ├── e06_reusable_state_assimilation.py
│   ├── e07_transformer_boundary.py
│   └── e08_claim_freeze.py
├── src/catena/
│   ├── data/      # geometry sweep, operator families, semantic records
│   ├── theory/    # constrained reachability, joint diagonalization
│   ├── models/    # matched scalar/semantic controllers
│   ├── training/  # matched and semantic probe trainers
│   ├── eval/      # raw metrics, seed inference, claims
│   └── systems/   # cost model and official adapters
├── configs/       # one YAML per experiment
├── docs/
├── tests/
└── papers/research_plan/
```

E01b was deliberately added instead of changing the running E01. E02-E08 are the joint scientific revision layer.
