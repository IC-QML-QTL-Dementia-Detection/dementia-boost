"""Dressed Quantum Network (DQN) classification head using Qiskit.

Composes the classical pre-net, angle scaling, Qiskit quantum layer, and
classical post-net into an interchangeable PyTorch `nn.Module`, mirroring the
PennyLane `QuantumClassifierHead` contract while executing the variational
circuit through Qiskit Primitives V2.
"""

import math

import torch.nn as nn
from qiskit.primitives import BaseEstimatorV2
from torch import Tensor, tanh

from .qiskit_circuit import DEFAULT_N_LAYERS, DEFAULT_N_QUBITS
from .qiskit_layer import DEFAULT_SPSA_EPSILON, create_qiskit_quantum_layer


class QiskitQuantumClassifierHead(nn.Module):
    """Dressed Quantum Network (DQN) classification head using Qiskit v2.x.

    Replaces classical dense layers with a parameterized quantum circuit
    flanked by a classical pre-net for dimensionality reduction and a
    classical post-net for logit mapping:
    1. Flattens input feature map: `(Batch, 64, 6, 6) -> (Batch, 2304)`
    2. Pre-Net: `Linear(2304 -> n_qubits)`
    3. Angle Scaling: `tanh(x) * (pi / 2)` mapping values into `[-pi/2, pi/2]`
    4. Variational Quantum Circuit: Evaluates expectation values `<PauliZ_i>`
       through an injectable Qiskit estimator, with SPSA gradients.
    5. Post-Net: `Linear(n_qubits -> 1)` producing raw classification logits.

    Attributes:
        DEFAULT_IN_FEATURES: Default number of incoming flattened features (2304).
        DEFAULT_N_QUBITS: Default number of qubits in the circuit (6).
        DEFAULT_N_LAYERS: Default number of Ansatz repetitions (4).
        DEFAULT_OUT_FEATURES: Default number of output classification units (1).
        ANGLE_SCALING_FACTOR: Multiplier scaling tanh outputs into `[-pi/2, pi/2]`.
        flatten: PyTorch Flatten layer.
        pre_net: Classical linear projection layer from `in_features` to `n_qubits`.
        qnn: `QiskitQuantumLayer` executing the Qiskit variational quantum circuit.
        post_net: Classical linear layer mapping qubit expectation values to logits.
    """

    DEFAULT_IN_FEATURES: int = 2304
    DEFAULT_N_QUBITS: int = DEFAULT_N_QUBITS
    DEFAULT_N_LAYERS: int = DEFAULT_N_LAYERS
    DEFAULT_OUT_FEATURES: int = 1
    ANGLE_SCALING_FACTOR: float = math.pi / 2.0

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        out_features: int = DEFAULT_OUT_FEATURES,
        spsa_epsilon: float = DEFAULT_SPSA_EPSILON,
        seed: int | None = None,
        estimator: BaseEstimatorV2 | None = None,
    ) -> None:
        """Initializes the hybrid Dressed Quantum Network classification head.

        Args:
            in_features: Number of incoming flattened features from CNN extractor.
                Defaults to 2304 (64 * 6 * 6).
            n_qubits: Number of qubits in the variational quantum circuit.
                Defaults to 6.
            n_layers: Number of repetitions (depth) of the ansatz. Defaults to 4.
            out_features: Number of output classification units. Defaults to 1.
            spsa_epsilon: Finite-difference step of the SPSA gradient estimate
                taken through the quantum layer. Defaults to
                `DEFAULT_SPSA_EPSILON`.
            seed: Seed of the SPSA perturbation generator. If None, follows the
                seed locked by `core.reproducibility.set_seed`.
            estimator: Optional Qiskit Primitives V2 estimator executing the
                circuit. If None, resolves to a noiseless `StatevectorEstimator`.
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.pre_net = nn.Linear(in_features=in_features, out_features=n_qubits)
        self.qnn = create_qiskit_quantum_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            spsa_epsilon=spsa_epsilon,
            seed=seed,
            estimator=estimator,
        )
        self.post_net = nn.Linear(in_features=n_qubits, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the Dressed Quantum Network.

        Args:
            x: Spatial feature map tensor or flattened embedding tensor of shape
                `(Batch, Channels, H, W)` or `(Batch, Features)`.

        Returns:
            Raw logit tensor of shape `(Batch, 1)`.
        """
        x = self.flatten(x)
        x = self.pre_net(x)
        x = tanh(x) * self.ANGLE_SCALING_FACTOR
        x = self.qnn(x)
        x = self.post_net(x)

        return x

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """Applies Glorot Uniform initialization to classical linear layers.

        Initializes weights of linear submodules (pre-net and post-net) using
        Xavier uniform distribution and resets biases to zero. The Qiskit
        circuit weights inside `qnn` remain untouched, preserving their
        uniform `[-pi, pi]` initialization.

        Args:
            module: The PyTorch submodule being visited during recursive
                initialization traversal.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
