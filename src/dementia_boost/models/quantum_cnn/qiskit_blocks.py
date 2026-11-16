"""Variational blocks: the trainable layers that follow the embedding.

A block defines its weights and one layer of gates; the circuit builder in
`qiskit_circuit.py` repeats the layer `n_layers` times after the embedding. The
weights of a block are a list of named `ParameterVector`s, and the quantum layer
stores them as one flat tensor in that order, so the order of the vectors is
part of a block's contract: saved checkpoints depend on it.

Registered blocks: `paper` (the base paper's block, the default), the
hardware-efficient block with linear, ring, or full CNOT entanglement, the
strongly entangling block, and a block with trainable ZZ couplings. Weights of
every block are initialised uniformly in `[-pi, pi]` by the quantum layer.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

from qiskit.circuit import ParameterVector, QuantumCircuit

from dementia_boost.core.identity import DEFAULT_BLOCK

__all__ = ["BLOCKS", "DEFAULT_BLOCK", "VariationalBlock", "get_block"]


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
    return _named_vectors({"theta": size, "gamma": size, "beta": size})


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


def _named_vectors(
    sizes: dict[str, int],
) -> list[ParameterVector]:
    """Creates one `ParameterVector` per name, in the given order.

    Args:
        sizes: Mapping of vector name to its length.

    Returns:
        The vectors, in the order of `sizes`.
    """
    return [ParameterVector(name, size) for name, size in sizes.items()]


def _cnot_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Lists the `(control, target)` pairs of a CNOT entangler.

    Args:
        n_qubits: Number of qubits.
        entanglement: `"linear"` for `i -> i+1`, `"ring"` for linear plus
            `n-1 -> 0` (only with more than two qubits, where it is a new
            pair), or `"full"` for `i -> j` with every `i < j`.

    Returns:
        The pairs, in the order the CNOTs are applied.

    Raises:
        ValueError: If `entanglement` is not known.
    """
    linear = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if entanglement == "linear":
        return linear
    if entanglement == "ring":
        return linear + [(n_qubits - 1, 0)] if n_qubits > 2 else linear
    if entanglement == "full":
        return list(itertools.combinations(range(n_qubits), 2))
    raise ValueError(
        f"Unknown entanglement {entanglement!r}; use 'linear', 'ring', or 'full'."
    )


def _apply_hardware_efficient_layer(
    circuit: QuantumCircuit,
    weight_vectors: list[ParameterVector],
    layer: int,
    entanglement: str,
) -> None:
    """Appends one hardware-efficient layer: rotations, then CNOTs.

    1. `RY(theta[layer, i])` on each qubit.
    2. `RZ(phi[layer, i])` on each qubit.
    3. CNOTs over the entanglement pattern (see `_cnot_pairs`).

    Args:
        circuit: Circuit to extend.
        weight_vectors: The `theta` and `phi` vectors.
        layer: Index of the layer.
        entanglement: `"linear"`, `"ring"`, or `"full"`.
    """
    theta, phi = weight_vectors
    n_qubits = circuit.num_qubits
    offset = layer * n_qubits

    for i in range(n_qubits):
        circuit.ry(theta[offset + i], i)

    for i in range(n_qubits):
        circuit.rz(phi[offset + i], i)

    for control, target in _cnot_pairs(n_qubits, entanglement):
        circuit.cx(control, target)


def _hardware_efficient_block(name: str, entanglement: str) -> VariationalBlock:
    """Builds a hardware-efficient block for one entanglement pattern.

    Args:
        name: Registry name.
        entanglement: `"linear"`, `"ring"`, or `"full"`.

    Returns:
        The block, with `2 * n_qubits` weights per layer.
    """
    return VariationalBlock(
        name=name,
        weight_vectors=lambda n_qubits, n_layers: _named_vectors(
            {"theta": n_layers * n_qubits, "phi": n_layers * n_qubits}
        ),
        apply_layer=lambda circuit, weight_vectors, layer: (
            _apply_hardware_efficient_layer(
                circuit, weight_vectors, layer, entanglement
            )
        ),
    )


def _strongly_entangling_weight_vectors(
    n_qubits: int, n_layers: int
) -> list[ParameterVector]:
    """Creates the `phi`, `theta`, and `omega` vectors of the general rotations.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of layers.

    Returns:
        Three vectors of `n_layers * n_qubits` parameters each.
    """
    size = n_layers * n_qubits
    return _named_vectors({"phi": size, "theta": size, "omega": size})


def _apply_strongly_entangling_layer(
    circuit: QuantumCircuit, weight_vectors: list[ParameterVector], layer: int
) -> None:
    """Appends one strongly entangling layer (Schuld et al., 2020).

    1. On each qubit, the general rotation `RZ(omega) RY(theta) RZ(phi)`
       (PennyLane's `Rot(phi, theta, omega)`): `RZ(phi)` is applied first.
    2. A ring of CNOTs `i -> (i + r) % n_qubits` with range
       `r = (layer % (n_qubits - 1)) + 1`, PennyLane's default rule for
       `StronglyEntanglingLayers`, so the range changes from layer to layer.

    Args:
        circuit: Circuit to extend.
        weight_vectors: The `phi`, `theta`, and `omega` vectors.
        layer: Index of the layer.

    Raises:
        ValueError: If the circuit has fewer than two qubits.
    """
    phi, theta, omega = weight_vectors
    n_qubits = circuit.num_qubits
    if n_qubits < 2:
        raise ValueError("The strongly entangling block needs at least two qubits.")
    offset = layer * n_qubits

    for i in range(n_qubits):
        circuit.rz(phi[offset + i], i)
        circuit.ry(theta[offset + i], i)
        circuit.rz(omega[offset + i], i)

    shift = (layer % (n_qubits - 1)) + 1
    for i in range(n_qubits):
        circuit.cx(i, (i + shift) % n_qubits)


def _trainable_zz_weight_vectors(n_qubits: int, n_layers: int) -> list[ParameterVector]:
    """Creates the coupling vector `w` and the rotation vector `alpha`.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of layers.

    Returns:
        `w` with `n_layers * (n_qubits - 1)` couplings and `alpha` with
        `n_layers * n_qubits` rotations.
    """
    return _named_vectors(
        {"w": n_layers * (n_qubits - 1), "alpha": n_layers * n_qubits}
    )


def _apply_trainable_zz_layer(
    circuit: QuantumCircuit, weight_vectors: list[ParameterVector], layer: int
) -> None:
    """Appends one layer with trainable ZZ couplings.

    1. `RZZ(w[layer, i])` on each neighbouring pair `(i, i + 1)`: the coupling
       of the ZZ feature map, with a trained angle instead of one computed
       from the inputs.
    2. `RX(alpha[layer, i])` on each qubit, which mixes the computational basis
       states that the diagonal couplings only give phases to.

    Args:
        circuit: Circuit to extend.
        weight_vectors: The `w` and `alpha` vectors.
        layer: Index of the layer.
    """
    couplings, rotations = weight_vectors
    n_qubits = circuit.num_qubits

    for i in range(n_qubits - 1):
        circuit.rzz(couplings[layer * (n_qubits - 1) + i], i, i + 1)

    for i in range(n_qubits):
        circuit.rx(rotations[layer * n_qubits + i], i)


BLOCKS: dict[str, VariationalBlock] = {
    DEFAULT_BLOCK: VariationalBlock(
        name=DEFAULT_BLOCK,
        weight_vectors=_paper_weight_vectors,
        apply_layer=_apply_paper_layer,
    ),
    "hardware_efficient_linear": _hardware_efficient_block(
        "hardware_efficient_linear", "linear"
    ),
    "hardware_efficient_ring": _hardware_efficient_block(
        "hardware_efficient_ring", "ring"
    ),
    "hardware_efficient_full": _hardware_efficient_block(
        "hardware_efficient_full", "full"
    ),
    "strongly_entangling": VariationalBlock(
        name="strongly_entangling",
        weight_vectors=_strongly_entangling_weight_vectors,
        apply_layer=_apply_strongly_entangling_layer,
    ),
    "trainable_zz": VariationalBlock(
        name="trainable_zz",
        weight_vectors=_trainable_zz_weight_vectors,
        apply_layer=_apply_trainable_zz_layer,
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
