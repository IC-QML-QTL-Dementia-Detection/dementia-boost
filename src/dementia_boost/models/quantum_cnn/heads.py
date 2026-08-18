"""Dressed Quantum Network (DQN) classification head for hybrid QTL models.

This module provides `QuantumClassifierHead`, which replaces classical dense
layers with a hybrid architecture comprising a classical pre-net linear projection,
trigonometric angle scaling, a variational quantum circuit, and a classical
post-net linear layer producing classification logits.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .circuit import create_quantum_layer


class QuantumClassifierHead(nn.Module):
    """Dressed Quantum Network (DQN) classification head for binary prediction.

    Replaces classical dense layers with a parameterized quantum circuit flanked
    by a classical pre-net for dimensionality reduction and a classical post-net
    for logit mapping:
    1. Flattens input feature map: `(Batch, 64, 6, 6) -> (Batch, 2304)`
    2. Pre-Net: `Linear(2304 -> n_qubits)`
    3. Angle Scaling: `tanh(x) * (pi / 2)` mapping values into `[-pi/2, pi/2]`
    4. Variational Quantum Circuit: Evaluates expectation values `<PauliZ_i>`
    5. Post-Net: `Linear(n_qubits -> 1)` producing raw classification logits.

    Attributes:
        flatten: PyTorch Flatten layer.
        pre_net: Classical linear projection layer from `in_features` to `n_qubits`.
        qnn: PennyLane TorchLayer executing the variational quantum circuit.
        post_net: Classical linear layer mapping qubit expectation values to logits.
    """

    def __init__(
        self,
        in_features: int = 2304,
        n_qubits: int = 6,
        n_layers: int = 4,
    ) -> None:
        """Initializes the hybrid Dressed Quantum Network classification head.

        Args:
            in_features: Number of incoming flattened features from CNN extractor.
                Defaults to 2304 (64 * 6 * 6).
            n_qubits: Number of qubits in the variational quantum circuit.
                Defaults to 6.
            n_layers: Number of repetitions (depth) of the ansatz. Defaults to 4.
        """
        super().__init__()

        self.flatten = nn.Flatten()

        self.pre_net = nn.Linear(in_features=in_features, out_features=n_qubits)

        self.qnn = create_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        self.post_net = nn.Linear(in_features=n_qubits, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the Dressed Quantum Network.

        Args:
            x: Spatial feature map tensor from the backbone of shape
                `(Batch, Channels, H, W)`.

        Returns:
            Raw logit tensor of shape `(Batch, 1)`.
        """
        x = self.flatten(x)

        x = self.pre_net(x)

        x = torch.tanh(x) * (math.pi / 2.0)

        x = self.qnn(x)

        x = self.post_net(x)

        return x

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """Applies Glorot Uniform initialization to classical linear layers.

        Initializes weights of linear submodules (pre-net and post-net) using
        Xavier uniform distribution and resets biases to zero.

        Args:
            module: The PyTorch submodule being visited during recursive
                initialization traversal.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
