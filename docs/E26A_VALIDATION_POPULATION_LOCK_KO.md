# E26a validation population lock

`configs/e26a_operator_data_gate.yaml`의
`gate_population.population_hash_required: true`는 E26a에서 실제로 읽는
validation population에 적용한다.

- seed: `260001`
- namespace: `e26a_gate_population_v1`
- domain: config에 고정된 4개 domain
- operation: config에 고정된 5개 operation
- episode 수: operation마다 100개
- query bundle: `current_state`, `derived_action`, `stale_probe`,
  `unaffected_retention`

Stage-2 protocol lock을 만들 때 위 validation episode만 두 번 독립적으로
재생성하고, 전체 canonical record와 `records_sha256`을
`E26A_VALIDATION_POPULATION_LOCK`에 고정한다. 이 lock의 파일 SHA-256은
protocol, numerical audit, restart audit가 모두 상속한다. E26a executor는
임의로 episode를 새로 선택하지 않고, lock과 config의 deterministic replay가
완전히 일치할 때만 ADD/INVALIDATE validation subset을 읽는다.

Config의 `splits`에 있는 `main_test`와 `heldout_domain`은 이후 stage가 사용할
수 있는 derivation specification일 뿐이다. E26a population-lock builder는
그 목록을 순회하지 않으며 다음을 명시적으로 기록한다.

```text
main_test_opened: false
main_test_access_count: 0
heldout_domain_opened: false
heldout_domain_access_count: 0
```

따라서 E26b protocol lock 전에는 main-test/held-out episode bytes를 생성,
평가 또는 tokenizer 입력으로 사용할 수 없다. Seed `260026`의 별도
transaction replay manifest는 training/calibration replay와 leakage audit용이며,
E26a validation outcome population을 대신하지 않는다.

