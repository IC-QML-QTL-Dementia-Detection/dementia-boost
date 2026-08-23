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
        DEFAULT_IN_FEATURES: Default number of flattened input features (2304).
        DEFAULT_HIDDEN_FEATURES: Default number of hidden units (5).
        DEFAULT_OUT_FEATURES: Default number of classification outputs (1).
        DEFAULT_DROPOUT_RATE: Default dropout probability (0.5).
        classifier: Sequential container of flattening, linear, dropout, and
            activation layers.
    """

    DEFAULT_IN_FEATURES: int = 2304
    DEFAULT_HIDDEN_FEATURES: int = 5
    DEFAULT_OUT_FEATURES: int = 1
    DEFAULT_DROPOUT_RATE: float = 0.5

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        out_features: int = DEFAULT_OUT_FEATURES,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        use_sigmoid: bool = True,
    ) -> None:
        """Initializes the classical classification head.

        Args:
            in_features: Number of flattened features from the extractor backbone.
                Defaults to 2304.
            hidden_features: Number of intermediate dense features. Defaults to 5.
            out_features: Number of output classification units. Defaults to 1.
            dropout_rate: Dropout probability during training. Defaults to 0.5.
            use_sigmoid: Whether to append a Sigmoid activation function to the
                final linear layer. Defaults to True.
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the dense classifier head.

        Args:
            x: Spatial feature map tensor or flattened embedding tensor of shape
                `(Batch, Channels, Height, Width)` or `(Batch, Features)`.

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
