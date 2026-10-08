"""Neural and hybrid quantum neural network model architectures and builders.

This module provides convolutional backbones, classical dense classification heads,
parameterized quantum circuit heads (Dressed Quantum Networks), and factory
builders for Classical Transfer Learning (CTL) and Quantum Transfer Learning (QTL).
"""

from .builder import (
    build_ctl_model,
    build_pl_qtl_model,
    build_qiskit_qtl_model,
)
from .classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from .quantum_cnn import PennylaneQuantumClassifierHead

__all__ = [
    "ClassicalClassifierHead",
    "DementiaClassifier",
    "LeNetFeatureExtractor",
    "PennylaneQuantumClassifierHead",
    "build_ctl_model",
    "build_pl_qtl_model",
    "build_qiskit_qtl_model",
]
