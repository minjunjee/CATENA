from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class OfficialBackendNotConfigured(RuntimeError):
    pass


@dataclass(slots=True)
class OfficialBackendConfig:
    repository: Path
    revision: str
    backend_name: str

    def validate(self) -> None:
        if not self.repository.exists():
            raise OfficialBackendNotConfigured(
                f"{self.backend_name} repository not found: {self.repository}"
            )
        if not self.revision:
            raise OfficialBackendNotConfigured(
                f"{self.backend_name} revision must be pinned explicitly."
            )


class KVEraserAdapter:
    def __init__(self, config: OfficialBackendConfig) -> None:
        self.config = config
        self.config.validate()

    def readiness(self) -> dict[str, str]:
        return {
            "backend": self.config.backend_name,
            "repository": str(self.config.repository),
            "revision": self.config.revision,
            "status": "adapter_contract_ready",
        }

    def erase(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "Scientific KVEraser execution must be wired to the pinned upstream repository."
        )
