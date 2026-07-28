# Post-Core 운영·개발 특기사항

## Repository와 storage

- Live repository는 `/home/minjun_dev/CATENA`를 유지했다.
- Artifact root는 `/data/minjun_dev/CATENA/artifacts`다.
- Repository의 `artifacts`는 위 large-storage 경로를 가리키는 symlink다.
- 기존 H1–H5 source/artifact를 교체하거나 삭제하지 않았다.

## Extension 설치

v6.2 post-core extension만 additive하게 설치했다. Full mock repository는
live repository 대체에 사용하지 않았다.

```text
extension_zip_sha256:
25efa443e732490c1ecaac3ee957925f8115f555fcf120a03c8de6438a23ec36
```

설치 manifest와 live overlay hash를 확인했고 신규 CPU dry workflow를
통과했다.

## Launcher safety

다음 운영 보강은 scientific config/metric과 독립적이다.

- exact `catena-v6` interpreter prefix 확인
- `python -m experiments...` 실행
- launcher flock
- live process duplicate guard
- completed target duplicate guard
- `/tmp/catena_postcore_dry*` 외 cleanup 거부
- symlink/mountpoint cleanup 거부

## E13b-R1 launcher 특기사항

첫 wave의 shell preflight는 통과했지만, 현재 Codex execution environment가
launcher shell 종료 시 `nohup` child process도 정리했다. 네 process는
run-start provenance만 기록하고 종료됐으며 traceback이나 GPU OOM은
없었다.

해당 manifest-only directory는 삭제·이동하지 않고 보존한다. E13c-R1은
이를 hash와 exclusion reason을 기록한 뒤 aggregate 입력에서 제외하고,
완전한 report/metrics/checkpoint를 가진 유일한 5-seed×2-variant run만
허용한다.

이 동작은 E13c 평가 전에
`E13C_R1_OPERATIONAL_INCOMPLETE_FILTER_AMENDMENT_LOCK.json`
(`13b12eb64bbe2806aaa3802bf8d908667dad48c127764c9f4edc56b49a3001cf`)
으로 고정했다. Config, metric, threshold와 statistical unit은 변경하지
않았다.

실제 training은 동일 locked command를 네 개의 장기 PTY session으로
실행했다. 이 변경은 config, seed, model, metric 또는 gate를 바꾸지 않는다.

## Official backend

Official repository, exact commit과 plugin이 설정되지 않아 실제 E15를
실행하지 않았다. Dry gate는 `official_backend_ready=false`로 정상
차단했으며 reference/mock fallback을 사용하지 않았다.
