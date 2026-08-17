"""Training, fine-tuning, and inference lifecycle runners.

This module provides `BaselineTrainer` for executing decoupled training and
evaluation loops with checkpointing, and `ModelEvaluator` for extracting
ground-truth labels and raw probabilities during dataset inference.
"""

from .evaluator import ModelEvaluator
from .trainer import BaselineTrainer

__all__ = [
    "BaselineTrainer",
    "ModelEvaluator",
]
