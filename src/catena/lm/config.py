from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Variant = Literal["dual_delta_lm", "projected_tied_delta_lm"]


def canonical_variant(value: str) -> Variant:
    aliases: dict[str, Variant] = {
        "dual": "dual_delta_lm",
        "dual_delta_lm": "dual_delta_lm",
        "tied": "projected_tied_delta_lm",
        "tied_delta_lm": "projected_tied_delta_lm",
        "projected_tied": "projected_tied_delta_lm",
        "projected_tied_delta_lm": "projected_tied_delta_lm",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unknown CATENA variant: {value!r}") from exc


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    ffn_multiplier: float = 4.0
    recurrent_layers: tuple[int, ...] = (0, 1, 2, 4, 5, 6)
    local_attention_layers: tuple[int, ...] = (3, 7)
    local_attention_window: int = 256
    context_length: int = 4096
    rms_norm_eps: float = 1.0e-5
    key_norm_eps: float = 1.0e-6
    dropout: float = 0.0
    weight_tying: bool = True
    reference_chunk_size: int = 32
    optimized_chunk_size: int = 32
    variant: Variant = "dual_delta_lm"
    backend_id: str = "reference_python"
    backend_scientific_main_capable: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        all_layers = set(self.recurrent_layers) | set(self.local_attention_layers)
        if all_layers != set(range(self.n_layers)):
            raise ValueError(
                "recurrent_layers and local_attention_layers must form an exact "
                "partition of range(n_layers)"
            )
        if set(self.recurrent_layers) & set(self.local_attention_layers):
            raise ValueError("recurrent and attention layer sets must be disjoint")
        if self.local_attention_window <= 0:
            raise ValueError("local_attention_window must be positive")
        if self.context_length <= 1:
            raise ValueError("context_length must exceed one token")
        if self.reference_chunk_size <= 0 or self.optimized_chunk_size <= 0:
            raise ValueError("reference and optimized chunk sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.backend_id not in {"reference_python", "compiled_scan"}:
            raise ValueError(f"Unsupported backend_id: {self.backend_id!r}")
        if self.backend_scientific_main_capable and self.backend_id == "reference_python":
            raise ValueError("reference_python cannot be marked scientific_main_capable")
        object.__setattr__(self, "variant", canonical_variant(self.variant))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        raw = int(self.d_model * self.ffn_multiplier)
        # Hardware-friendly multiple while preserving a deterministic config.
        return ((raw + 63) // 64) * 64

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["recurrent_layers"] = list(self.recurrent_layers)
        result["local_attention_layers"] = list(self.local_attention_layers)
        result["head_dim"] = self.head_dim
        result["ffn_hidden_dim"] = self.ffn_hidden_dim
        return result

    @classmethod
    def tiny_reference(cls, variant: str = "dual_delta_lm") -> ModelConfig:
        return cls(
            vocab_size=259,
            n_layers=2,
            d_model=32,
            n_heads=4,
            ffn_multiplier=2.0,
            recurrent_layers=(0, 1),
            local_attention_layers=(),
            local_attention_window=16,
            context_length=64,
            reference_chunk_size=8,
            optimized_chunk_size=8,
            variant=canonical_variant(variant),
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ModelConfig:
        data = dict(mapping)
        data.pop("head_dim", None)
        data.pop("ffn_hidden_dim", None)
        if "recurrent_layers" in data:
            data["recurrent_layers"] = tuple(int(x) for x in data["recurrent_layers"])
        if "local_attention_layers" in data:
            data["local_attention_layers"] = tuple(int(x) for x in data["local_attention_layers"])
        if "variant" in data:
            data["variant"] = canonical_variant(str(data["variant"]))
        return cls(**data)


@dataclass(frozen=True)
class RunSafetyConfig:
    dry_run_root_prefix: str = "/tmp/catena_e26_dry_"
    canonical_artifact_root: str = "/data/minjun_dev/CATENA/artifacts"
    require_allow_main: bool = True
    forbid_reference_backend_for_main: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    experiment: str
    stage: str
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Config must be a mapping: {source}")
        for key in ("schema_version", "experiment", "stage"):
            if key not in payload:
                raise ValueError(f"Missing required config key {key!r}: {source}")
        if payload["schema_version"] != "catena-v8.1":
            raise ValueError(
                f"Expected schema_version=catena-v8.1, got {payload['schema_version']!r}"
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            experiment=str(payload["experiment"]),
            stage=str(payload["stage"]),
            raw=payload,
        )
