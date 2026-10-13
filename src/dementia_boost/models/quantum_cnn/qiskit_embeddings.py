"""Embeddings (feature maps) that load classical features into the Qiskit circuit.

An embedding has two halves that must agree. The angle function, written in
PyTorch, turns the scaled pre-net output of shape `(Batch, n_qubits)` into the
gate angles of shape `(Batch, n_features)`. The circuit part places the gates
that consume those angles, one circuit parameter per angle. Computing every
angle outside the circuit keeps each circuit parameter a plain input column, so
the circuit never multiplies run-time parameters, and autograd differentiates
the angle function exactly back to the pre-net.

The variational block that follows the embedding lives in `qiskit_circuit.py`
and is the same for every embedding.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from qiskit.circuit import ParameterVector, QuantumCircuit
from torch import Tensor

DEFAULT_EMBEDDING: str = "paper"
ZZ_REPETITIONS: int = 2


@dataclass(frozen=True)
class Embedding:
    """An embedding: its angle function, its width, and its circuit part.

    Attributes:
        name: Registry name, recorded as the `ansatz` of a run specification.
        n_features: Maps the number of qubits to the number of gate angles.
        angles: Maps scaled inputs `(Batch, n_qubits)` to gate angles
            `(Batch, n_features)`.
        apply: Appends the embedding gates to a circuit, consuming the angle
            parameters in the column order of `angles`.
    """

    name: str
    n_features: Callable[[int], int]
    angles: Callable[[Tensor], Tensor]
    apply: Callable[[QuantumCircuit, ParameterVector], None]


def _identity_angles(x: Tensor) -> Tensor:
    """Returns the inputs unchanged, one angle per qubit.

    Args:
        x: Scaled inputs of shape `(Batch, n_qubits)`.

    Returns:
        The same tensor.
    """
    return x


def _one_angle_per_qubit(n_qubits: int) -> int:
    """Returns the number of angles of an embedding with one angle per qubit.

    Args:
        n_qubits: Number of qubits.

    Returns:
        `n_qubits`.
    """
    return n_qubits


def _apply_angle_embedding(circuit: QuantumCircuit, params: ParameterVector) -> None:
    """Appends a Hadamard layer and `RZ(x_i)` on every qubit.

    This is the embedding of the base paper plus the Hadamard layer it needs:
    without it the state stays at `|0...0>` and the circuit is constant.

    Args:
        circuit: Circuit to extend.
        params: One angle parameter per qubit.
    """
    for qubit in range(circuit.num_qubits):
        circuit.h(qubit)
    for qubit in range(circuit.num_qubits):
        circuit.rz(params[qubit], qubit)


def _apply_rx_embedding(circuit: QuantumCircuit, params: ParameterVector) -> None:
    """Appends `RX(x_i)` on every qubit, without a Hadamard layer.

    Args:
        circuit: Circuit to extend.
        params: One angle parameter per qubit.
    """
    for qubit in range(circuit.num_qubits):
        circuit.rx(params[qubit], qubit)


def _pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Lists the qubit pairs coupled by a ZZ feature map.

    Args:
        n_qubits: Number of qubits.
        entanglement: `"full"` for every pair, or `"linear"` for nearest
            neighbours `(i, i + 1)`, as in Qiskit's `ZZFeatureMap`.

    Returns:
        The pairs `(i, j)` with `i < j`, in the order of the angle columns.

    Raises:
        ValueError: If `entanglement` is not known.
    """
    if entanglement == "full":
        return list(itertools.combinations(range(n_qubits), 2))
    if entanglement == "linear":
        return [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    raise ValueError(f"Unknown entanglement {entanglement!r}; use 'full' or 'linear'.")


def _zz_angles(x: Tensor, entanglement: str) -> Tensor:
    """Computes the gate angles of the ZZ feature map.

    Args:
        x: Scaled inputs of shape `(Batch, n_qubits)`.
        entanglement: `"full"` or `"linear"`.

    Returns:
        Tensor of shape `(Batch, n_qubits + n_pairs)`: the single-qubit angles
        `2 x_i`, then the pair angles `2 (pi - x_i)(pi - x_j)`.
    """
    pairs = _pairs(x.shape[1], entanglement)
    first = torch.tensor([i for i, _ in pairs], device=x.device)
    second = torch.tensor([j for _, j in pairs], device=x.device)
    pair_angles = 2.0 * (math.pi - x[:, first]) * (math.pi - x[:, second])
    return torch.cat([2.0 * x, pair_angles], dim=1)


def _apply_zz_embedding(
    circuit: QuantumCircuit, params: ParameterVector, entanglement: str
) -> None:
    """Appends the ZZ feature map: repeated blocks of Hadamards and phases.

    Each repetition applies a Hadamard on every qubit, `RZ(2 x_i)` on every
    qubit, and `RZZ` on every coupled pair, as in Qiskit's `ZZFeatureMap`. `RZ`
    equals its `P` gate and `RZZ` equals its CNOT, `P`, CNOT sequence, both up
    to a global phase, so every expectation value is the same.

    Args:
        circuit: Circuit to extend.
        params: Single-qubit angle parameters followed by pair angle parameters.
        entanglement: `"full"` or `"linear"`.
    """
    n_qubits = circuit.num_qubits
    pairs = _pairs(n_qubits, entanglement)
    for _ in range(ZZ_REPETITIONS):
        for qubit in range(n_qubits):
            circuit.h(qubit)
        for qubit in range(n_qubits):
            circuit.rz(params[qubit], qubit)
        for index, (first, second) in enumerate(pairs):
            circuit.rzz(params[n_qubits + index], first, second)


def _zz_embedding(name: str, entanglement: str) -> Embedding:
    """Builds a ZZ feature map embedding for one entanglement pattern.

    Args:
        name: Registry name.
        entanglement: `"full"` or `"linear"`.

    Returns:
        The embedding.
    """
    return Embedding(
        name=name,
        n_features=lambda n_qubits: n_qubits + len(_pairs(n_qubits, entanglement)),
        angles=lambda x: _zz_angles(x, entanglement),
        apply=lambda circuit, params: _apply_zz_embedding(
            circuit, params, entanglement
        ),
    )


EMBEDDINGS: dict[str, Embedding] = {
    DEFAULT_EMBEDDING: Embedding(
        name=DEFAULT_EMBEDDING,
        n_features=_one_angle_per_qubit,
        angles=_identity_angles,
        apply=_apply_angle_embedding,
    ),
    "rx": Embedding(
        name="rx",
        n_features=_one_angle_per_qubit,
        angles=_identity_angles,
        apply=_apply_rx_embedding,
    ),
    "zz_feature_map": _zz_embedding("zz_feature_map", "full"),
    "zz_feature_map_linear": _zz_embedding("zz_feature_map_linear", "linear"),
}


def get_embedding(name: str) -> Embedding:
    """Looks an embedding up by its registry name.

    Args:
        name: Registry name, such as `"paper"` or `"zz_feature_map"`.

    Returns:
        The registered embedding.

    Raises:
        ValueError: If no embedding has that name.
    """
    try:
        return EMBEDDINGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding {name!r}; available: {sorted(EMBEDDINGS)}."
        ) from None
