"""PennyLane variational quantum circuit (VQC) ansatz and PyTorch layer bridge.

This module constructs the custom variational ansatz described by Bhowmik et al.
(2025), implementing angle embedding via RZ rotations, an entangling ring of CNOT
gates, parameterized RZ and controlled-RY gates, and Pauli-Z expectation value
measurements wrapped in a PyTorch `TorchLayer`.
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.measurements import ExpectationMP
from pennylane.qnn.torch import TorchLayer

DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 4
PARAMETERS_PER_LAYER: int = 3


def create_quantum_layer(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
) -> TorchLayer:
    """Factory function instantiating a PennyLane QNode as a PyTorch Module.

    Attempts to attach high-performance state-vector backends (`lightning.gpu`,
    `lightning.qubit`) before falling back to `default.qubit`. Configures the
    QNode with adjoint differentiation and uniform parameter initialization
    over `[-pi, pi]`.

    Args:
        n_qubits: Number of qubits in the circuit. Defaults to 6.
        n_layers: Number of Ansatz layer repetitions. Defaults to 4.

    Returns:
        A PyTorch-compatible TorchLayer executing the quantum circuit.
    """
    try:
        device = qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        try:
            device = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch", diff_method="adjoint")
    def qnode(inputs: torch.Tensor, weights: torch.Tensor) -> list[ExpectationMP]:
        return _build_custom_ansatz(inputs, weights, n_qubits, n_layers)

    weight_shapes = {"weights": (n_layers, PARAMETERS_PER_LAYER, n_qubits)}
    init_method = {
        "weights": lambda tensor: nn.init.uniform_(tensor, a=-torch.pi, b=torch.pi)
    }

    return TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)


def _build_custom_ansatz(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
) -> list[ExpectationMP]:
    """Constructs the parameterized quantum circuit ansatz.

    Implements initial state preparation using RZ angle embedding of the scaled
    classical input features, followed by `n_layers` repetitions of:
    1. Parameterized RZ rotation on each qubit: `RZ(weights[layer, 0, i])`
    2. Entangling ring of CNOT gates between adjacent qubits: `CNOT(i, (i + 1) % n)`
    3. Second parameterized RZ rotation on each qubit: `RZ(weights[layer, 1, i])`
    4. Entangling controlled-RY gate:
       `CRY(weights[layer, 2, i], wires=[(i + 1) % n, i])`

    Finally measures the expectation value of the Pauli-Z observable on every
    qubit.

    Args:
        inputs: Scaled classical feature tensor of shape `(Batch, n_qubits)`
            bounded in `[-pi/2, pi/2]`.
        weights: Trainable parameter tensor of shape `(n_layers, 3, n_qubits)`.
        n_qubits: Number of qubits in the quantum circuit.
        n_layers: Number of ansatz layer repetitions.

    Returns:
        A list of PennyLane Pauli-Z expectation measurements for all qubits.
    """
    for i in range(n_qubits):
        qml.RZ(inputs[:, i], wires=i)  # type: ignore

    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[layer, 0, i], wires=i)  # type: ignore

        for i in range(n_qubits):
            target = (i + 1) % n_qubits
            qml.CNOT(wires=[i, target])

        for i in range(n_qubits):
            qml.RZ(weights[layer, 1, i], wires=i)  # type: ignore

        for i in range(n_qubits):
            control = (i + 1) % n_qubits
            target = i
            qml.CRY(weights[layer, 2, target], wires=[control, target])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
