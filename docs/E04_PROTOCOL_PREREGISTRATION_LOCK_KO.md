# E04 protocol preregistration lock

- Frozen protocol: `docs/E04_PROTOCOL_PREREGISTRATION_FROZEN_KO.md`
- Frozen at: `2026-07-27T05:31:57.093945Z`
- Protocol SHA-256: `06e98f7c1449181cc98be051d7dfc9a90ec6d66b93dae9de14d8f9da34f2a4c9`
- Config canonical SHA-256: `8db019198e4c8a46f29a83c57a5eed4646d841da840ff67a45f997bde8854f1b`
- Config byte SHA-256: `f6d7a18c2ec831c8847236de46f443056dedbfc68198123fa82d104615fbc5ea`
- E04 checkpoint intervention calls before lock: `0`
- E04 run directories present before lock: `0`
- E04 main namespace inspected before lock: `false`
- E04 dry namespace inspected before lock: `false`

Pre-evaluation implementation validation에서 independent-donor raw recovery에
전용 bootstrap seed가 빠진 clerical omission을 발견했다. 결과 계산이나
namespace 열람 전에 seed `421`을 추가하고 이 lock을 다시 고정했다.

E04 runner는 protocol, lock, config, E02b repair, 원본 E02 checkpoint contract를
intervention evaluation 전에 검증한다. 같은 source tree는 이어지는 새 E00에서
다시 fingerprint한다.
