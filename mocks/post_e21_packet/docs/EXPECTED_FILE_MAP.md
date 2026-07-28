# Reference-only expected file map

Codex should adapt names to the live E18–E21 convention, but the logical deliverables are:

```text
experiments/
  e22a_locality_method_selection.py
  e22b_active_path_locality.py
  e23a_product_poset_screen.py
  e23b_product_poset_confirmatory.py
  e24a_approximate_rank_stress.py
  e24b_behavioral_attainability_stress.py
  e25a_official_gdn2_gate.py
  e25b_text_transaction_anchor.py
configs/
  matching YAML files
docs/
  E22_ACTIVE_PATH_LOCALITY_PROTOCOL_KO.md
  E23_PRODUCT_POSET_PROTOCOL_KO.md
  E24_THEORY_STRESS_PROTOCOL_KO.md
  E25A_OFFICIAL_GDN2_PROTOCOL_KO.md
  E25B_TEXT_TRANSACTION_PROTOCOL_KO.md
  POST_E21_DEPENDENCY_DAG_KO.md
  POST_E21_IMPLEMENTATION_REPORT_KO.md
scripts/
  launch_post_e21_wave1.sh
  check_post_e21_status.sh
```

Do not create a second experiment tree if the live repo uses `src/catena/experiments` instead.
