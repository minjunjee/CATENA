# E26 Stage-2 data-tool environment amendment

## 발견된 구현 결함

Scientific data builder는 `catena-v6`과 분리된 pinned virtual environment를
사용하도록 등록했지만, `catena.lm` package와 `catena.lm.hashing`이 import
시점에 PyTorch를 즉시 불러왔다. 따라서 PyTorch를 설치하지 않은 data-only
environment에서 schedule builder가 다음과 같이 중단됐다.

```text
ModuleNotFoundError: No module named 'torch'
```

이어 `catena-v6`에서 schedule builder를 직접 호출한 진단에서는 pinned
`tokenizers` package가 없으므로 다음과 같이 fail-closed했다.

```text
ScientificTokenizerContractError:
Scientific tokenizer runtime requires the optional 'tokenizers' package
```

두 실패 모두 scientific run 이전의 dependency validation이며 canonical E26
artifact를 만들지 않았다. Data split, tokenizer input, model outcome, threshold와
seed는 보지 않았고 변경하지 않았다.

## Prospective repair

- `catena.lm`의 torch-backed public model export를 PEP 562 lazy attribute로
  전환했다. 기존 `from catena.lm import CatenaLM` API는 유지한다.
- `catena.lm.hashing`은 tensor/model digest 함수가 실제 호출될 때만 PyTorch를
  import한다. Pure JSON/SHA helper는 data-only environment에서 독립적으로
  사용할 수 있다.
- data venv는 `--system-site-packages` 없이 생성한다.
- construction-source receipt는 이 두 transitive dependency와
  `transactional_stream.py`, 공용 provenance helper까지 SHA-256으로 고정한다.

수정 후 isolated environment에서 PyTorch가 import되지 않은 상태로
transaction replay와 paired schedule replay를 재생성했다. 두 결과는 기존
결과와 byte-for-byte 동일했다.

```text
transaction manifest SHA-256:
71849b23f58da7fbaf4c1e835e7f33cb2b5f12d03a8c038a445dbe51d51449e6

schedule manifest SHA-256:
19d04449aa5fd0c1f5f675fe3768ca7ceef98c79908b6d5d4c0ceeeaf735b8f5
```

이 amendment는 environment isolation과 provenance completeness만 수리한다.
Scientific E26a, E26b, E26c는 실행하지 않는다.
