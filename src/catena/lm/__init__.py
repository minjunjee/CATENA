"""CATENA v8.1 reference LM contracts.

This package deliberately contains a slow reference recurrence for correctness,
unit tests, and non-evidence smoke runs. Scientific MAIN runs must provide an
optimized backend whose manifest declares ``scientific_main_capable=true``.
"""

from typing import TYPE_CHECKING, Any

from .config import ModelConfig, canonical_variant

if TYPE_CHECKING:
    from .model import CatenaLM, RuntimeState, build_paired_models

__all__ = [
    "CatenaLM",
    "ModelConfig",
    "RuntimeState",
    "build_paired_models",
    "canonical_variant",
]

__version__ = "8.1.0"


def __getattr__(name: str) -> Any:
    """Load torch-backed model objects only when callers request them.

    Corpus/tokenizer construction runs in a deliberately isolated environment
    that does not install PyTorch. Keeping the public model API lazy lets those
    data-only modules import without coupling scientific input construction to
    the training environment.
    """

    if name in {"CatenaLM", "RuntimeState", "build_paired_models"}:
        from .model import CatenaLM, RuntimeState, build_paired_models

        exported = {
            "CatenaLM": CatenaLM,
            "RuntimeState": RuntimeState,
            "build_paired_models": build_paired_models,
        }
        globals().update(exported)
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
