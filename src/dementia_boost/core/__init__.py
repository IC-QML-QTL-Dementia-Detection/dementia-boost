"""Core module providing reproducibility, run identity, and runtime utilities.

This module centralizes deterministic random seed management across Python,
NumPy, PyTorch, and cuDNN backend engines, and the typed run specification with
the hash IDs and label derived from it.
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
from .reproducibility import set_seed

__all__ = [
    "Paradigm",
    "RunSpec",
    "backbone_id_of",
    "config_id",
    "label",
    "run_id",
    "set_seed",
    "short_hash",
]
