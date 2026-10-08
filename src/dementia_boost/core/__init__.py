"""Core module providing reproducibility, run identity, and runtime utilities.

This module centralizes deterministic random seed management across Python,
NumPy, PyTorch, and cuDNN backend engines, the typed run specification with the
hash IDs and label derived from it, and the on-disk layout of run artifacts.
"""

from .identity import (
    Paradigm,
    RunSpec,
    backbone_id_of,
    config_id,
    label,
    run_id,
    short_hash,
)
from .layout import ConfigCollisionError, ResultsLayout
from .reproducibility import set_seed

__all__ = [
    "ConfigCollisionError",
    "Paradigm",
    "ResultsLayout",
    "RunSpec",
    "backbone_id_of",
    "config_id",
    "label",
    "run_id",
    "set_seed",
    "short_hash",
]
