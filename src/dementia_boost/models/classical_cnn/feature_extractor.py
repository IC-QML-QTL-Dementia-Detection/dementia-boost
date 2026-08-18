"""Convolutional spatial feature extractor based on a modified LeNet backbone.

This module provides `LeNetFeatureExtractor`, which processes 2D grayscale brain
MRI slices of shape (Batch, 1, 128, 128) through progressive convolutional and
pooling blocks to produce high-level spatial feature maps of shape (Batch, 64, 6, 6).
"""

import torch.nn as nn
from torch import Tensor


class LeNetFeatureExtractor(nn.Module):
    """Convolutional backbone extracting spatial representations from MRI slices.

    The architecture comprises four convolutional stages with ReLU activations
    and max pooling, downsampling (1, 128, 128) inputs to a (64, 6, 6) feature
    map (2304 flattened features).

    Attributes:
        features: The sequential container of convolutional, activation, and
            pooling layers.
    """

    def __init__(self) -> None:
        """Initializes the LeNet convolutional backbone layers."""
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the convolutional backbone.

        Args:
            x: Input tensor of shape `(Batch, 1, 128, 128)`.

        Returns:
            Extracted feature map tensor of shape `(Batch, 64, 6, 6)`.
        """
        return self.features(x)
