# CATENA v6.2 Post-Core Extension

이 배포본은 현재 live CATENA v6.1 repository에 **새 파일만 추가**한다. 완료된 H1-H5 source, config, artifact와 report는 수정하지 않는다. 단, Codex가 새 단계의 규칙을 자동으로 발견할 수 있도록 root `CODEX.md`와 `AGENTS.md` 끝에 범위가 표시된 addendum만 추가한다.

## 포함 범위

- E10 learned control-rank scaling
- E11 representation-control co-adaptation
- E12 architecture-demand control lattice
- E13a/b/c structured transactional event-sequence bridge
- E14 structured plan-state continuation
- E15 pinned official-backend gate
- E16 completed-core evidence freeze
- 실험별 Python entry point, YAML config, 한국어 protocol 문서와 claim gate
- 4-GPU launcher, dependency-aware launcher, status tool, full CPU dry-run
- v6.2 보강 연구계획서 PDF/LaTeX

## 설치

```bash
cd /home/minjun_dev
rm -rf CATENA_v62_extension
unzip -q CATENA_v6.2_postcore_extension.zip -d CATENA_v62_extension

source /home/minjun_dev/miniconda3/bin/activate catena-v6

CATENA_PYTHON="$(command -v python)" \
  bash /home/minjun_dev/CATENA_v62_extension/CATENA_v6.2_postcore_extension/APPLY_EXTENSION.sh \
  /home/minjun_dev/CATENA
```

설치기는 다음을 수행한다.

1. 기존 경로와 충돌하는 새 파일이 있으면 중단
2. 새 파일만 복사
3. `CODEX.md`/`AGENTS.md`에 idempotent한 post-core addendum 추가
4. compileall, 새 unit test, shell syntax 검사
5. E10-E16 전체 CPU dry-run
6. 실패 시 복사 파일 제거와 instruction file rollback

## 설치 후 확인

```bash
cd /home/minjun_dev/CATENA
set -a; source .env; set +a

bash scripts/check_postcore_status.sh \
  /home/minjun_dev/CATENA \
  /data/minjun_dev/CATENA/artifacts

bash scripts/run_postcore_dry.sh \
  /home/minjun_dev/CATENA \
  /tmp/catena_postcore_dry_manual
```

## 즉시 실행

```bash
bash scripts/launch_postcore_wave1.sh /home/minjun_dev/CATENA

python experiments/e16_core_evidence_freeze.py \
  --config configs/e16_core_evidence_freeze.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

세부 dependency와 결과 해석은 `docs/NEXT_ACTIONS_KO.md`와 `docs/POSTCORE_DEPENDENCY_DAG_KO.md`를 따른다.
