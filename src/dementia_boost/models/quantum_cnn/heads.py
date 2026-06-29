import math

import torch
import torch.nn as nn
from torch import Tensor

from .circuit import create_quantum_layer


class QuantumClassifierHead(nn.Module):
    """
    A Dressed Quantum Network (DQN) classification head for binary prediction.

    Replaces classical dense layers with a parameterized quantum circuit, flanked by
    a classical pre-net for dimensionality reduction and a post-net for logit mapping.
    """

    def __init__(
        self,
        in_features: int = 2304,
        n_qubits: int = 6,
        n_layers: int = 4,
    ):
        """
        Initializes the hybrid classification head.

        Args:
            in_features (int): Features from the flattened CNN output. Defaults to 2304.
            n_qubits (int): Number of qubits in the VQC. Defaults to 6.
            n_layers (int): Number of repetitions in the ansatz. Defaults to 4.
        """
        super().__init__()

        self.flatten = nn.Flatten()

        self.pre_net = nn.Linear(in_features=in_features, out_features=n_qubits)

        self.qnn = create_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        self.post_net = nn.Linear(in_features=n_qubits, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the forward pass through the Dressed Quantum Network.

        Args:
            x (Tensor): The 2D feature map from the extractor.
                Shape: (Batch, Channels, H, W)

        Returns:
            Tensor: Raw logits of shape (Batch, 1).
        """
        x = self.flatten(x)

        x = self.pre_net(x)

        x = torch.tanh(x) * (math.pi / 2.0)

        x = self.qnn(x)

        x = self.post_net(x)

        return x

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """
        Applies Glorot (Xavier) Uniform initialization to the classical linear layers
        (pre-net and post-net) and zeroes their biases.

        Args:
            module (nn.Module): The current PyTorch submodule being evaluated.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
