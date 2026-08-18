"""Classical fully connected classification heads for dementia detection.

This module provides `ClassicalClassifierHead`, which flattens incoming spatial
feature maps, applies dense layers with dropout regularization and ReLU activations,
and produces binary classification outputs.
"""

import torch.nn as nn
from torch import Tensor


class ClassicalClassifierHead(nn.Module):
    """Classical dense classification head for binary prediction.

    Flattens spatial feature maps, passes them through a dense bottleneck
    (2304 -> 5) with dropout regularization (p=0.5) and ReLU activation,
    and projects down to a single output unit (5 -> 1).

    Attributes:
        classifier: Sequential container of flattening, linear, dropout, and
            activation layers.
    """

    def __init__(self, in_features: int = 2304, use_sigmoid: bool = True) -> None:
        """Initializes the classical classification head.

        Args:
            in_features: The number of flattened features from the extractor
                backbone. Defaults to 2304 (64 * 6 * 6).
            use_sigmoid: Whether to append a Sigmoid activation function to the
                final linear layer. Defaults to True.
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=5),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=5, out_features=1),
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the dense classifier head.

        Args:
            x: Spatial feature map tensor from the backbone of shape
                `(Batch, Channels, Height, Width)`.

        Returns:
            Output prediction tensor of shape `(Batch, 1)`.
        """
        return self.classifier(x)

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """Applies Glorot (Xavier) Uniform weight initialization.

        Initializes weights of linear submodules with Xavier uniform
        distribution and resets biases to zero. Designed for use with
        PyTorch's `model.apply()`.

        Args:
            module: The PyTorch submodule being visited during recursive
                initialization traversal.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
