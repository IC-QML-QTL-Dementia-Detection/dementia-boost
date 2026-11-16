"""Qiskit parameterized variational circuit for Dressed Quantum Networks.

Composes a registered embedding (`qiskit_embeddings.py`) and `n_layers` layers
of a registered variational block (`qiskit_blocks.py`) into one Qiskit v2.x
`QuantumCircuit`, with `SparsePauliOp` Pauli-Z observables. With the defaults it
is the circuit of Bhowmik et al. (2025) plus a Hadamard layer, the counterpart
of the PennyLane circuit in `pennylane_circuit.py`.
"""

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .qiskit_blocks import DEFAULT_BLOCK, get_block
from .qiskit_embeddings import DEFAULT_EMBEDDING, get_embedding

DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 4


def build_qiskit_ansatz(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    embedding: str = DEFAULT_EMBEDDING,
    block: str = DEFAULT_BLOCK,
) -> tuple[
    QuantumCircuit,
    ParameterVector,
    list[ParameterVector],
    list[SparsePauliOp],
]:
    """Constructs the parameterized Qiskit circuit and Pauli-Z observables.

    Builds the chosen embedding of the classical input features (by default a
    Hadamard layer followed by an RZ angle embedding, see `qiskit_embeddings`),
    then `n_layers` layers of the chosen variational block (by default the
    Bhowmik et al. (2025) block, see `qiskit_blocks`).

    The Hadamard layer is a deliberate departure from the circuit drawn in the
    paper. Without it the state stays at `|0...0>` and every expectation value
    is 1 for all inputs and weights.

    Pauli-Z observables are built with `SparsePauliOp` strings reversed
    relative to qubit index, since Qiskit orders Pauli strings from the most
    significant (last) to the least significant (first) qubit.

    Args:
        n_qubits: Number of qubits in the circuit. Defaults to 6.
        n_layers: Number of variational block layers. Defaults to 4.
        embedding: Registry name of the embedding. Defaults to the paper's
            angle embedding.
        block: Registry name of the variational block. Defaults to the
            paper's block.

    Returns:
        A tuple containing:
            - circuit: The parameterized QuantumCircuit.
            - input_params: ParameterVector for the embedding's gate angles
              (x), one per column of the embedding's angle function.
            - weight_params: The block's weight ParameterVectors, in the order
              of the flat weight tensor (`[theta, gamma, beta]` for `paper`).

    Raises:
        ValueError: If the embedding or the block is not registered.
            - observables: List of SparsePauliOp Pauli-Z operators, one per
              qubit.
    """
    circuit = QuantumCircuit(n_qubits)

    chosen_embedding = get_embedding(embedding)
    input_params = ParameterVector("x", chosen_embedding.n_features(n_qubits))
    chosen_embedding.apply(circuit, input_params)

    chosen_block = get_block(block)
    weight_vectors = chosen_block.weight_vectors(n_qubits, n_layers)
    for layer in range(n_layers):
        chosen_block.apply_layer(circuit, weight_vectors, layer)

    observables = []
    for i in range(n_qubits):
        pauli_str = ["I"] * n_qubits
        pauli_str[n_qubits - 1 - i] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1.0)]))

    return circuit, input_params, weight_vectors, observables
