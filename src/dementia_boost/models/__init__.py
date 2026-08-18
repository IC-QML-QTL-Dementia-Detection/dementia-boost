"""Neural and hybrid quantum neural network model architectures and builders.

This module provides convolutional backbones, classical dense classification heads,
parameterized quantum circuit heads (Dressed Quantum Networks), and factory
builders for Classical Transfer Learning (CTL) and Quantum Transfer Learning (QTL).
"""

from .builder import build_classical_tl_model, build_quantum_tl_model
from .classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from .quantum_cnn import QuantumClassifierHead

__all__ = [
    "ClassicalClassifierHead",
    "DementiaClassifier",
    "LeNetFeatureExtractor",
    "QuantumClassifierHead",
    "build_classical_tl_model",
    "build_quantum_tl_model",
]
