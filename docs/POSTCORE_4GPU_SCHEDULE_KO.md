# 4-GPU 실행 일정

## Wave 1 - 즉시 병렬 실행

| GPU | 실험 | 의존성 | 종료 후 판단 |
|---:|---|---|---|
| 0 | E10 learned rank scaling | 없음 | learned rank가 reachable rank floor를 추적하는지 |
| 1 | E11 representation-control co-adaptation | 없음 | common rotation 회복 및 noncommuting residual |
| 2 | E12 architecture-demand lattice | 없음 | 각 자유도의 선택적 이득과 simpler-task non-inferiority |
| 3 | E13a-R1 paired sequence floor/throughput | 없음 | E13b GO/NO-GO 및 공정한 forward ETA |

```bash
bash scripts/launch_postcore_wave1.sh /home/minjun_dev/CATENA
```

## CPU lane - 동시에 실행

```bash
python experiments/e16_core_evidence_freeze.py   --config configs/e16_core_evidence_freeze.yaml   --device cpu   --artifact-root /data/minjun_dev/CATENA/artifacts
```

## Wave 2 - prospective E13a-R1 GO 후

```bash
bash scripts/launch_sequence_if_go.sh /home/minjun_dev/CATENA
# wave 1 완료 후
bash scripts/launch_sequence_wave.sh /home/minjun_dev/CATENA 2
# wave 2 완료 후
bash scripts/launch_sequence_wave.sh /home/minjun_dev/CATENA 3
```

## Aggregate 및 plan continuation

```bash
python experiments/e13c_transactional_sequence_aggregate.py   --config configs/e13c_transactional_sequence_aggregate.yaml   --device cpu   --artifact-root /data/minjun_dev/CATENA/artifacts

python experiments/e14_plan_continuation.py   --config configs/e14_plan_continuation.yaml   --device cuda:0   --artifact-root /data/minjun_dev/CATENA/artifacts
```

Repaired E13c-R1이 `MAIN/PASS/SUPPORTED` claim gate를 열지 않으면 E14는
실행하지 않는다. E14는 E13c-R1 provenance가 봉인한 다섯 dual E13b-R1
checkpoint의 full seed×update×gap grid를 평가하며, arbitrary checkpoint
override는 없다.
