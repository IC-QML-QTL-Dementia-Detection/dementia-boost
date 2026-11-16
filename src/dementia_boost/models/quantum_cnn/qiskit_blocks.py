"""Variational blocks: the trainable layers that follow the embedding.

A block defines its weights and one layer of gates; the circuit builder in
`qiskit_circuit.py` repeats the layer `n_layers` times after the embedding. The
weights of a block are a list of named `ParameterVector`s, and the quantum layer
stores them as one flat tensor in that order, so the order of the vectors is
part of a block's contract: saved checkpoints depend on it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from qiskit.circuit import ParameterVector, QuantumCircuit

DEFAULT_BLOCK: str = "paper"


@dataclass(frozen=True)
class VariationalBlock:
    """A variational block: its weights and the gates of one layer.

    Attributes:
        name: Registry name, recorded as the `block` of a run specification.
        weight_vectors: Maps `(n_qubits, n_layers)` to the block's weight
            vectors, in the order of the flat weight tensor.
        apply_layer: Appends one layer of gates to a circuit, given the weight
            vectors and the layer index.
    """

    name: str
    weight_vectors: Callable[[int, int], list[ParameterVector]]
    apply_layer: Callable[[QuantumCircuit, list[ParameterVector], int], None]


def _paper_weight_vectors(n_qubits: int, n_layers: int) -> list[ParameterVector]:
    """Creates the `theta`, `gamma`, and `beta` vectors of the paper's block.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of layers.

    Returns:
        Three vectors of `n_layers * n_qubits` parameters each.
    """
    size = n_layers * n_qubits
    return [ParameterVector(name, size) for name in ("theta", "gamma", "beta")]


def _apply_paper_layer(
    circuit: QuantumCircuit, weight_vectors: list[ParameterVector], layer: int
) -> None:
    """Appends one layer of the Bhowmik et al. (2025) block.

    1. `RZ(theta[layer, i])` on each qubit.
    2. A ring of CNOTs, `CNOT(i, (i + 1) % n_qubits)`.
    3. `RZ(gamma[layer, i])` on each qubit.
    4. A ring of controlled RY gates, `CRY(beta[layer, i])` with control
       `(i + 1) % n_qubits` and target `i`.

    Args:
        circuit: Circuit to extend.
        weight_vectors: The `theta`, `gamma`, and `beta` vectors.
        layer: Index of the layer.
    """
    theta, gamma, beta = weight_vectors
    n_qubits = circuit.num_qubits
    offset = layer * n_qubits

    for i in range(n_qubits):
        circuit.rz(theta[offset + i], i)

    for i in range(n_qubits):
        circuit.cx(i, (i + 1) % n_qubits)

    for i in range(n_qubits):
        circuit.rz(gamma[offset + i], i)

    for i in range(n_qubits):
        circuit.cry(beta[offset + i], (i + 1) % n_qubits, i)


BLOCKS: dict[str, VariationalBlock] = {
    DEFAULT_BLOCK: VariationalBlock(
        name=DEFAULT_BLOCK,
        weight_vectors=_paper_weight_vectors,
        apply_layer=_apply_paper_layer,
    ),
}


def get_block(name: str) -> VariationalBlock:
    """Looks a variational block up by its registry name.

    Args:
        name: Registry name, such as `"paper"`.

    Returns:
        The registered block.

    Raises:
        ValueError: If no block has that name.
    """
    try:
        return BLOCKS[name]
    except KeyError:
        raise ValueError(
            f"Unknown variational block {name!r}; available: {sorted(BLOCKS)}."
        ) from None
