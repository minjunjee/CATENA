# E26a 실행 준비 판정

## 현재 판정

```text
optimized_candidate_smoke: PASS
control_protocol_lock: PASS
scientific_data_bundle: MISSING
complete_e26a_gate_runner: NOT_CONNECTED
execution_readiness: BLOCKED_DEPENDENCY
scientific_main_started: false
```

통과한 smoke는 batch 1, sequence 256, random-token 100-step
`NON_EVIDENCE_VALIDATION`이다. 따라서 context 4096의 real 80/20 stream,
20M-token floor pilot 또는 E26c 일정의 근거로 직접 사용하면 안 된다.

## 고정된 control input

| 항목 | 경로 / SHA-256 |
|---|---|
| Source capture | `docs/E26A_SOURCE_CAPTURE.json` / `c55b28bc83cd5b09e9b5165005ef37c3b3cfc4d0de86e94219a83ed66a8661d7` |
| Protocol lock | `docs/E26A_OPERATOR_DATA_GATE_PROTOCOL_LOCK.json` / `76483a53706e25ec346809fdbff79548bdd69bf6d59c4ea643c5e51a7cd87f6e` |
| Config | `configs/e26a_operator_data_gate.yaml` / `0359337b4c1d152be97f7e7880f52dc86c348567e3c8d9f14caf47eab7ce3378` |
| Backend candidate | repair smoke `backend_manifest.json` / `822221639c79abd232500b8f35de887d0c0256d21aec2f50dc5285f68eb05adc` |

Control-only readiness validator는 PASS했다. Backend manifest는
`e26a_candidate_capable=true`이지만 의도대로
`e26a_gate_capable=false`, `parity_verified=false`,
`scientific_main_capable=false`를 유지한다.

## 남은 STOP 조건

1. exact 16K BPE/Unigram tokenizer model과 training-document manifest 없음
2. frozen real general-corpus document manifest와 token memmap 없음
3. optional `tokenizers` runtime이 `catena-v6`에 없음
4. batch 1/2/4, registered sequence/split/state/variant 전체 parity grid 미실행
5. 80/20 loader, fixed gate population, leakage/exact-refresh audit,
   최대 20M-token floor/headroom pilot, actual-context throughput projection을
   수행하는 non-dry E26a runner가 아직 연결되지 않음

이 중 1–3은 외부 data/toolchain 선택과 설치 승인이 필요하다. 4–5는 다음
implementation amendment에서 outcome을 보지 않고 완료해야 한다.

## 승인 후 입력 검증 명령

아래 명령은 이미 준비된 외부 파일을 검증할 뿐 다운로드나 tokenizer training을
하지 않는다.

```bash
cd /home/minjun_dev/CATENA_E26
source /home/minjun_dev/miniconda3/bin/activate catena-v6
export PYTHONPATH=/home/minjun_dev/CATENA_E26/src

python tools/materialize_e26_data.py \
  --tokenizer-manifest /data/minjun_dev/CATENA/e26_inputs/v8_1/tokenizer_manifest.json \
  --corpus-manifest /data/minjun_dev/CATENA/e26_inputs/v8_1/general_corpus_manifest.json \
  --output /data/minjun_dev/CATENA/e26_inputs/v8_1/scientific_data_readiness.json \
  --sequence-length 4096

python tools/check_e26a_readiness.py \
  --repo-root /home/minjun_dev/CATENA_E26 \
  --config configs/e26a_operator_data_gate.yaml \
  --protocol-lock docs/E26A_OPERATOR_DATA_GATE_PROTOCOL_LOCK.json \
  --backend-manifest \
    /tmp/catena_e26_dry_gpu_smoke_repair_20260731T0716Z/e26a_operator_data_gate/20260731T071132.954822Z/backend_manifest.json \
  --tokenizer-manifest /data/minjun_dev/CATENA/e26_inputs/v8_1/tokenizer_manifest.json \
  --corpus-manifest /data/minjun_dev/CATENA/e26_inputs/v8_1/general_corpus_manifest.json \
  --output /data/minjun_dev/CATENA/e26_inputs/v8_1/e26a_execution_readiness.json
```

E26a runner 연결, full validation, 위 receipt PASS, 사용자 승인 이후에만 launcher를
사용한다.

```bash
CATENA_EXECUTE_MAIN=YES_I_HAVE_APPROVED \
BACKEND_MANIFEST=<full-e26a-candidate-manifest.json> \
PROTOCOL_LOCK=/home/minjun_dev/CATENA_E26/docs/E26A_OPERATOR_DATA_GATE_PROTOCOL_LOCK.json \
TOKENIZER_MANIFEST=/data/minjun_dev/CATENA/e26_inputs/v8_1/tokenizer_manifest.json \
CORPUS_MANIFEST=/data/minjun_dev/CATENA/e26_inputs/v8_1/general_corpus_manifest.json \
bash /home/minjun_dev/CATENA_E26/scripts/launch_e26a_2gpu.sh
```

현재 이 launcher를 실행하면 fail-closed로 중단되는 것이 정상이다.

## Feasibility scenario

Smoke rate `1,963.16 token/s/model`을 단순 외삽하면 250M token은
35.37 GPU-hour/model, 등록 safety factor 1.25 적용 시 44.22 GPU-hour/model이다.
10개 run을 4 GPU의 3 wave로 배치한 단순 상한은 약 132.65시간(5.53일)이다.

이는 sequence 256에서의 scenario일 뿐 ETA lock이 아니다. E26a는 context
4096, 실제 batch와 loader에서 다시 측정한 뒤 outcome과 무관하게
250M–500M 범위의 token budget을 잠가야 한다.

