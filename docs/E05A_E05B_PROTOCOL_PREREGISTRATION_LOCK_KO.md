# E05a/E05b protocol preregistration lock

- Frozen protocol:
  `docs/E05A_E05B_PROTOCOL_PREREGISTRATION_FROZEN_KO.md`
- Frozen at: `2026-07-27T07:21:03.269460Z`
- Protocol SHA-256:
  `e235a351ca84589d92a211e78f6f4ebe6631ce228adad8f8da34dec17527b8e0`
- E05a config canonical SHA-256:
  `4d95f6afa16ea66488125825a75f97831677a563c0bf3b2f9a09934535e637e7`
- E05a config byte SHA-256:
  `2895bb250f92744e0238d56fcdb85816584195ea6d2aba5a767899377a1ec6c2`
- E05b config canonical SHA-256:
  `c7c37a67c978b3cd2e3cf93a54669f57ebed86a6ce59e75194d59d6003b9d7ef`
- E05b config byte SHA-256:
  `67eb6d5efecbb1dc653efc388d8e0d18e386498644f3b6594e63f8d8fdea4eb8`
- E04 additive freeze SHA-256:
  `6d225b673da998cef9131af0b2d49fc699f89af2159f40c302898144c2765b30`
- New E05 dry/main run directories before lock: `0`
- New E05 model training/evaluation calls before lock: `0`
- E05a dry namespace inspected before lock: `false`
- E05a pilot namespace inspected before lock: `false`
- E05b train/validation/main/secondary namespace inspected before lock: `false`
- Legacy E05 outcome reused: `false`

Runner는 protocol, lock, 두 config, E04 freeze, fresh E00 source fingerprint를
model/data generation 전에 검증한다. E05a dry는 development namespace만 만들며
E05b registry를 생성하지 않는다. E05a main이 모든 design-validity gate를
통과한 경우에만 E05b registry와 300-item audit sample을 생성한다.

Human audit는 두 사람의 독립 review와 300행 adjudication을 별도 additive
artifact로 기록한다. AI agent가 human review를 대신 채우지 않는다. Audit
dependency가 통과하기 전 E05b training을 시작하지 않는다.
