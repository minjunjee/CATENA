# E15 - Official Backend Gate

## 원칙

Reference recurrence와 official GDN2/KDA/KVEraser evidence를 분리한다. Official claim은 다음이 모두 충족될 때만 열린다.

1. upstream repo path 존재
2. full commit SHA 일치
3. 별도 plugin module 존재
4. FP32/BF16 numerical parity
5. tied-GDN2/KDA equivalence 또는 KVEraser full-recompute agreement
6. no reference fallback

## 환경

기존 `catena-v6`을 오염시키지 않는다. Official GDN2와 KVEraser는 별도 Conda 환경 또는 container로 설치한다.

## Plugin contract

`plugin_module`은 다음 함수를 노출한다.

```python
def run_backend_gate(config: dict) -> dict:
    return {
        "passed": True,
        "scientific_evidence": True,
        "metrics": {...},
    }
```

Plugin이 없거나 commit이 다르면 `NOT_CONFIGURED`로 기록하고 claim을 차단한다.
