"""
Classical Convolutional Neural Network module for dementia classification.
Exposes the LeNet feature extractor, the classical dense head, and the orchestrator.
"""

from .classifier import DementiaClassifier
from .feature_extractor import LeNetFeatureExtractor
from .heads import ClassicalClassifierHead

__all__ = [
    "DementiaClassifier",
    "LeNetFeatureExtractor",
    "ClassicalClassifierHead",
]
