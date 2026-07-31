"""CATENA v8.1 reference LM contracts.

This package deliberately contains a slow reference recurrence for correctness,
unit tests, and non-evidence smoke runs. Scientific MAIN runs must provide an
optimized backend whose manifest declares ``scientific_main_capable=true``.
"""

from .config import ModelConfig, canonical_variant
from .model import CatenaLM, RuntimeState, build_paired_models

__all__ = [
    "CatenaLM",
    "ModelConfig",
    "RuntimeState",
    "build_paired_models",
    "canonical_variant",
]

__version__ = "8.1.0"
