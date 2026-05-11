import torch.nn as nn
from torch import Tensor


class LeNetFeatureExtractor(nn.Module):
    """
    Extracts spatial features from 2D grayscale brain MRI images using a modified
    LeNet architecture.

    The network progressively downsamples the spatial dimensions while increasing
    the channel depth to extract high-level representations of brain tissue.
    Expects input tensors of shape [Batch, 1, 128, 128].
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # Block 4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass through the convolutional feature extractor.

        Args:
            x (Tensor): Input tensor of shape (Batch, 1, 128, 128).

        Returns:
            Tensor: Feature map of shape (Batch, 64, 6, 6).
        """
        return self.features(x)
