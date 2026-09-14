"""PyTorch bridge for Qiskit circuits via qiskit-machine-learning TorchConnector.

Wraps the Qiskit ansatz inside an `EstimatorQNN` evaluated by Qiskit's
`StatevectorEstimator` Primitive (V2) and exposes it as a PyTorch `nn.Module`
through `TorchConnector`, with autograd support for both circuit weights and
upstream classical inputs.
"""

import torch
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import BaseEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .qiskit_circuit import DEFAULT_N_LAYERS, DEFAULT_N_QUBITS, build_qiskit_ansatz


def create_qiskit_quantum_layer(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    estimator: BaseEstimatorV2 | None = None,
    gradient: BaseEstimatorGradient | None = None,
) -> TorchConnector:
    """Instantiates a Qiskit EstimatorQNN wrapped in a PyTorch TorchConnector.

    Builds the Bhowmik et al. (2025) ansatz, binds it to Pauli-Z observables,
    and evaluates expectation values through Qiskit Primitives V2. Input
    gradients are enabled so autograd can backpropagate through the classical
    pre-net feeding this layer, matching the differentiability guarantees of
    the PennyLane `TorchLayer` counterpart. Primitive precision is pinned to
    0.0 (exact, noiseless expectation values) so the ideal state-vector
    simulation is not perturbed by Primitives V2's default finite-shot noise
    model.

    Args:
        n_qubits: Number of qubits in the circuit. Defaults to 6.
        n_layers: Number of ansatz layer repetitions. Defaults to 4.
        estimator: Optional Qiskit Primitives V2 estimator instance. If None,
            defaults to a noiseless `StatevectorEstimator`.
        gradient: Optional gradient estimator strategy. If None, `EstimatorQNN`
            defaults to exact parameter-shift, whose cost scales linearly with
            the number of differentiable parameters (weights and inputs) and
            becomes impractical at the default 6-qubit, 4-layer, `n_layers`
            depth. Pass a `SPSAEstimatorGradient` for training workloads,
            which evaluates only 2 circuits per gradient step regardless of
            parameter count at the expense of exactness.

    Returns:
        A TorchConnector instance executing the circuit on the given
        estimator, with weights uniformly initialized over `[-pi, pi]`.
    """
    circuit, input_params, weight_param_vectors, observables = build_qiskit_ansatz(
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    all_weight_params = [param for vector in weight_param_vectors for param in vector]

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=list(input_params),
        weight_params=all_weight_params,
        estimator=estimator if estimator is not None else StatevectorEstimator(),
        gradient=gradient,
        input_gradients=True,
        default_precision=0.0,
    )

    initial_weights = (
        torch.rand(len(all_weight_params), dtype=torch.float32) * 2 * torch.pi
        - torch.pi
    ).numpy()

    return TorchConnector(qnn, initial_weights=initial_weights)
