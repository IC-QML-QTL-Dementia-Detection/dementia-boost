import torch.nn as nn
from torch import Tensor


class ClassicalClassifierHead(nn.Module):
    """
    A classical fully connected classification head for binary prediction.

    This module takes a flattened feature representation, passes it through dense
    layers with dropout for regularization, and outputs a single probability
    indicating the likelihood of dementia.
    """

    def __init__(self, in_features: int) -> None:
        """
        Initializes the classification head.

        Args:
            in_features (int): The number of features received from the flattened
                output of the feature extractor. Defaults to 2304 (64 * 6 * 6).
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=5),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass through the linear classifier.

        Args:
            x (Tensor): The 2D feature map from the extractor.
                Shape: (Batch, Channels, Height, Width)

        Returns:
            Tensor: A probability vector of shape (Batch, 1) bounded between [0.0, 1.0].
        """
        return self.classifier(x)
