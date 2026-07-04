import pennylane as qml
import torch
import torch.nn as nn
from pennylane.measurements import ExpectationMP
from pennylane.qnn.torch import TorchLayer


def create_quantum_layer(
    n_qubits: int = 6,
    n_layers: int = 4,
) -> TorchLayer:
    """
    Factory function to instantiate the PennyLane QNode and wrap it as a PyTorch Module.
    Uses the 'default.qubit' ideal simulator.

    Args:
        n_qubits (int): Number of qubits. Defaults to 6.
        n_layers (int): Number of Ansatz repetitions. Defaults to 4.

    Returns:
        TorchLayer: A PyTorch-compatible NN layer executing the quantum circuit.
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

    weight_shapes = {"weights": (n_layers, 2, n_qubits)}
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
    """
    Builds the custom quantum circuit used in the base paper.

    Args:
        inputs (torch.Tensor): The scaled classical data mapped to [-pi/2, pi/2].
        weights (torch.Tensor): The trainable parameters, shape (n_layers, 2, n_qubits).
        n_qubits (int): The number of qubits in the circuit.
        n_layers (int): The number of Ansatz repetitions.

    Returns:
        list[ExpectationMP]: A list of expectation values for all qubits.
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
            target = (i + 1) % n_qubits
            qml.CRY(weights[layer, 1, i], wires=[i, target])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
