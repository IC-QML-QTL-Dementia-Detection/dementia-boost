"""Classical Convolutional Neural Network components for dementia classification.

This subpackage exposes the LeNet spatial feature extractor backbone, the
classical dense classification head, and the dependency-injected
DementiaClassifier orchestrator.
"""

from .classifier import DementiaClassifier
from .feature_extractor import LeNetFeatureExtractor
from .heads import ClassicalClassifierHead

__all__ = [
    "ClassicalClassifierHead",
    "DementiaClassifier",
    "LeNetFeatureExtractor",
]
