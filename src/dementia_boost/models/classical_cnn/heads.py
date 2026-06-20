import torch.nn as nn
from torch import Tensor


class ClassicalClassifierHead(nn.Module):
    """
    A classical fully connected classification head for binary prediction.

    This module takes a flattened feature representation, passes it through dense
    layers with dropout for regularization, and outputs a single probability
    indicating the likelihood of dementia.
    """

    def __init__(self, in_features: int = 2304, use_sigmoid: bool = True) -> None:
        """
        Initializes the classification head.

        Args:
            in_features (int): The number of features received from the flattened
                output of the feature extractor. Defaults to 2304 (64 * 6 * 6).

            use_sigmoid (bool): Whether there should be a Sigmoid activation function
                or not.
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
        """
        Computes the forward pass through the linear classifier.

        Args:
            x (Tensor): The 2D feature map from the extractor.
                Shape: (Batch, Channels, Height, Width)

        Returns:
            Tensor: A probability vector of shape (Batch, 1) bounded between [0.0, 1.0].
        """
        return self.classifier(x)

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """
        Applies Glorot (Xavier) Uniform initialization to the linear layers
        and zeroes the biases.
        Designed to be passed into PyTorch's native model.apply() method.

        Args:
            module (nn.Module): The current PyTorch submodule being evaluated.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
