"""Qiskit parameterized variational circuit ansatz for Dressed Quantum Networks.

Constructs the Hadamard layer, angle embedding, and 3-parameter ansatz (theta,
gamma, beta) described by Bhowmik et al. (2025) using Qiskit v2.x `QuantumCircuit`,
`ParameterVector`, and `SparsePauliOp` observables. This is an independent,
interchangeable counterpart to the PennyLane ansatz in `pennylane_circuit.py`, sharing
the same mathematical formulation but executed through Qiskit Primitives.
"""

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 4


def build_qiskit_ansatz(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
) -> tuple[
    QuantumCircuit,
    ParameterVector,
    list[ParameterVector],
    list[SparsePauliOp],
]:
    """Constructs the parameterized Qiskit circuit and Pauli-Z observables.

    Builds an initial Hadamard layer followed by an RZ angle embedding of the
    classical input features, then `n_layers` repetitions of the Bhowmik et
    al. (2025) ansatz:
    1. Parameterized RZ rotation on each qubit: `RZ(theta[layer, i])`.
    2. Entangling ring of CNOT gates between adjacent qubits:
       `CNOT(i, (i + 1) % n_qubits)`.
    3. Second parameterized RZ rotation on each qubit: `RZ(gamma[layer, i])`.
    4. Entangling controlled-RY gate:
       `CRY(beta[layer, i], control=(i + 1) % n_qubits, target=i)`.

    The Hadamard layer is a deliberate departure from the circuit drawn in the
    paper. Without it the state stays at `|0...0>` and every expectation value
    is 1 for all inputs and weights.

    Pauli-Z observables are built with `SparsePauliOp` strings reversed
    relative to qubit index, since Qiskit orders Pauli strings from the most
    significant (last) to the least significant (first) qubit.

    Args:
        n_qubits: Number of qubits in the circuit. Defaults to 6.
        n_layers: Number of ansatz layer repetitions. Defaults to 4.

    Returns:
        A tuple containing:
            - circuit: The parameterized QuantumCircuit.
            - input_params: ParameterVector for classical angle inputs (x).
            - weight_params: List of `[theta, gamma, beta]` ParameterVectors.
            - observables: List of SparsePauliOp Pauli-Z operators, one per
              qubit.
    """
    circuit = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        circuit.h(i)

    input_params = ParameterVector("x", n_qubits)
    for i in range(n_qubits):
        circuit.rz(input_params[i], i)

    theta_params = ParameterVector("theta", n_layers * n_qubits)
    gamma_params = ParameterVector("gamma", n_layers * n_qubits)
    beta_params = ParameterVector("beta", n_layers * n_qubits)

    for layer in range(n_layers):
        offset = layer * n_qubits

        for i in range(n_qubits):
            circuit.rz(theta_params[offset + i], i)

        for i in range(n_qubits):
            target = (i + 1) % n_qubits
            circuit.cx(i, target)

        for i in range(n_qubits):
            circuit.rz(gamma_params[offset + i], i)

        for i in range(n_qubits):
            control = (i + 1) % n_qubits
            target = i
            circuit.cry(beta_params[offset + i], control, target)

    observables = []
    for i in range(n_qubits):
        pauli_str = ["I"] * n_qubits
        pauli_str[n_qubits - 1 - i] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1.0)]))

    return circuit, input_params, [theta_params, gamma_params, beta_params], observables
