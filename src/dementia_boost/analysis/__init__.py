"""Analyses that read circuits or results without training anything.

- `trainability`: the trainability of each circuit at initialisation.
"""

from .trainability import (
    CircuitChoice,
    TrainabilityResult,
    default_choices,
    diagnose,
    markdown_table,
)

__all__ = [
    "CircuitChoice",
    "TrainabilityResult",
    "default_choices",
    "diagnose",
    "markdown_table",
]
