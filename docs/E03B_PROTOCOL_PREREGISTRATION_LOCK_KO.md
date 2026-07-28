# E03b protocol preregistration lock

- Frozen protocol: `docs/E03B_PROTOCOL_PREREGISTRATION_FROZEN_KO.md`
- Frozen at: `2026-07-26T17:58:12.858533Z`
- Protocol SHA-256: `f1665900785509ca97c814395aee081574078f26ae9d8dd0a90bae4a0a5a15e6`
- Analytic pilot SHA-256: `21d60d79738e5bd05034312ec630a694d6485f00be1bc824344088eebc9fe94c`
- E03b empirical probe calls before lock: `0`
- E03b run directories present before lock: `0`
- Main candidate namespace inspected before lock: `false`
- Dry candidate namespace inspected before lock: `false`

E03b runner는 protocol, lock, pilot, config, candidate registry와 preserved E03
source hash를 empirical probe 전에 검증한다. 이 상태의 전체 source tree는
이어지는 새 E00 run에서 다시 fingerprint한다.
