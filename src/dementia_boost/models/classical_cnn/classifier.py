"""Orchestrator module combining feature extractor backbones with classification heads.

This module provides the `DementiaClassifier` container, which utilizes dependency
injection to decouple spatial representation learning from decision heads, enabling
seamless swapping between classical dense heads and quantum circuit heads.
"""

import torch.nn as nn
from torch import Tensor


class DementiaClassifier(nn.Module):
    """Dependency-injected orchestrator connecting a backbone to a classifier head.

    Decouples spatial feature extraction from classification logic, allowing
    different heads (such as classical dense layers or Dressed Quantum Networks)
    to be attached to the same pre-trained backbone.

    Attributes:
        feature_extractor: Neural network module extracting spatial features.
        classifier_head: Classification module mapping features to predictions.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier_head: nn.Module,
    ) -> None:
        """Initializes the composed classifier model.

        Args:
            feature_extractor: Network backbone responsible for extracting
                spatial features from the raw image tensor.
            classifier_head: Classification network responsible for mapping
                extracted features to predictions.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

    def forward(self, x: Tensor) -> Tensor:
        """Computes the full forward pass through the backbone and head.

        Args:
            x: Input image tensor of shape `(Batch, Channels, Height, Width)`.

        Returns:
            Output prediction tensor from the classification head.
        """
        features = self.feature_extractor(x)
        prediction = self.classifier_head(features)

        return prediction
