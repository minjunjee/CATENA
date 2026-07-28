# E13 - Transactional Event-Sequence Memory

## 역할

H1-H4는 analytic finite-memory probe에 가깝다. E13은 shared semantic encoder, repeated update, distractor gap을 포함하는 structured event sequence에서 factorization 원리가 유지되는지 확인한다.

## 입력

- entity ID
- old/new value ID
- operation label이 아닌 6개 raw relation field
- verified/unverified event bit
- 1, 4, 8회 update
- 0, 128, 512, 2048 distractor event

Address와 candidate는 아직 structured oracle이다. 따라서 natural language나 learned addressing claim은 열지 않는다.

## 모델

두 조건은 같은 encoder와 two-output head를 사용한다.

- tied: two outputs를 평균해 erase=write로 projection
- dual: erase와 write를 독립 사용

## E13a-R1 gate

- tied/dual paired initialization·training/evaluation contract
- dual affected-entity exact-match와 affected-MSE floor
- tied-dual affected-MSE gap
- unaffected retention
- warm-up 후 forward-only repeated throughput
- E13b 실제 scale의 짧은 training-step probe에서 투영한 run/wave ETA

하나라도 실패하면 E13b를 시작하지 않는다.

원본 E13a calibration artifact는 immutable pilot로 남기며 E13b dependency로
사용하지 않는다. 상세 수리는 `E13A_R1_SEQUENCE_CALIBRATION_KO.md`를 따른다.

## E13b/E13c

E13b는 5개 고정 seed에서 variant/seed별 checkpoint와 완전한 update×gap
grid를 생성한다. E13c는 각 variant×seed에 유일한 eligible run이 있는지,
source config와 checkpoint/report/metric hash가 일치하는지를 먼저 검증한 뒤
paired aggregate하여 seed-level sign-flip과 retention non-inferiority를
판정한다. 3 seeds에서는 one-sided exact sign-flip의 최소 p-value가
`0.125`이므로 confirmatory gate를 열 수 없다.

## 허용 주장

성공 시 structured event-sequence에서 repeated update factorization이 유지된다고 주장한다. Recurrent LM, agent, natural language claim은 금지한다.
