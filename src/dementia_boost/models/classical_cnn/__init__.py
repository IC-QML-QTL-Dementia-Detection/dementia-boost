"""Classical Convolutional Neural Network components for dementia classification.

This subpackage exposes the LeNet spatial feature extractor backbone, the
registry of classical classification heads, and the dependency-injected
DementiaClassifier orchestrator.
"""

from .classifier import DementiaClassifier
from .feature_extractor import LeNetFeatureExtractor
from .heads import (
    CLASSICAL_HEADS,
    ClassicalClassifierHead,
    TanhBottleneckHead,
    build_classical_head,
    classical_head_for,
    stored_hidden_features,
)

__all__ = [
    "CLASSICAL_HEADS",
    "ClassicalClassifierHead",
    "DementiaClassifier",
    "LeNetFeatureExtractor",
    "TanhBottleneckHead",
    "build_classical_head",
    "classical_head_for",
    "stored_hidden_features",
]
