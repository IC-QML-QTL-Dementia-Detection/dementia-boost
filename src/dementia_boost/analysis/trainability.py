"""Trainability of the quantum circuits at initialisation, without training.

For one circuit (an embedding, a variational block, and re-uploading on or
off), weights are drawn uniformly in `[-pi, pi]` and scaled features uniformly
in `[-pi/2, pi/2]`, as at the start of training, and the circuit is simulated
exactly. Two costs are read from the Pauli-Z expectation values: the mean of
all `<Z_i>` (global) and `<Z_0>` alone (local); barren-plateau results differ
between global and local costs. Reported per circuit:

- input spread: how much each `<Z_i>` varies across inputs at fixed weights,
  the constant-circuit guard as a number;
- input sensitivity: root mean square of the derivative of each cost with
  respect to the scaled features;
- weight-gradient scale and variance: root mean square of the derivative of
  each cost with respect to every circuit weight, and its variance across
  random initialisations (the usual barren-plateau measure);
- the number of input angles and weights, and their sum, the dimension that
  SPSA perturbs.

Derivatives are central differences on the exact simulation, all shifted
circuits evaluated in one batched estimator call. Controlled rotations such as
CRY have no two-term parameter-shift rule, so finite differences keep one method
for every block.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from qiskit.primitives import BaseEstimatorV2
from torch import Tensor

from dementia_boost.models.quantum_cnn import (
    BLOCKS,
    DEFAULT_BLOCK,
    DEFAULT_EMBEDDING,
    EMBEDDINGS,
    QiskitExpectationRunner,
    build_qiskit_ansatz,
    get_embedding,
)

DEFAULT_STEP: float = 1e-4
DEFAULT_QUBITS: tuple[int, ...] = (4, 6, 8, 10)
DEFAULT_LAYERS: int = 4
DEFAULT_WEIGHT_DRAWS: int = 20
DEFAULT_INPUTS: int = 10


@dataclass(frozen=True)
class CircuitChoice:
    """The choices that define a circuit, as in a run specification.

    Attributes:
        embedding: Registry name of the embedding.
        block: Registry name of the variational block.
        reupload: Whether the embedding is repeated before every layer.
    """

    embedding: str
    block: str
    reupload: bool


@dataclass(frozen=True)
class TrainabilityResult:
    """The diagnostic of one circuit at one size.

    Attributes:
        embedding: Registry name of the embedding.
        block: Registry name of the variational block.
        reupload: Whether the embedding is repeated before every layer.
        n_qubits: Number of qubits.
        n_layers: Number of block layers.
        n_input_angles: Number of input angles the embedding feeds the circuit.
        n_weights: Number of circuit weights.
        spsa_dimension: Number of values SPSA perturbs (angles plus weights).
        input_spread: Mean over weight draws of the smallest per-qubit standard
            deviation of `<Z_i>` across inputs.
        input_sensitivity_global: Root mean square of `d mean<Z> / d z_j`.
        input_sensitivity_local: Root mean square of `d <Z_0> / d z_j`.
        weight_gradient_rms_global: Root mean square of `d mean<Z> / d w_k`.
        weight_gradient_rms_local: Root mean square of `d <Z_0> / d w_k`.
        weight_gradient_variance_global: Variance of `d mean<Z> / d w_k` across
            weight draws, averaged over the weights.
        weight_gradient_variance_local: The same for `<Z_0>`.
    """

    embedding: str
    block: str
    reupload: bool
    n_qubits: int
    n_layers: int
    n_input_angles: int
    n_weights: int
    spsa_dimension: int
    input_spread: float
    input_sensitivity_global: float
    input_sensitivity_local: float
    weight_gradient_rms_global: float
    weight_gradient_rms_local: float
    weight_gradient_variance_global: float
    weight_gradient_variance_local: float

    def to_dict(self) -> dict[str, Any]:
        """Serialises the result.

        Returns:
            A JSON-serialisable dictionary of every field.
        """
        return asdict(self)


def central_differences(
    function: Callable[[Tensor], Tensor],
    points: Tensor,
    columns: Sequence[int] | None = None,
    step: float = DEFAULT_STEP,
) -> Tensor:
    """Differentiates a batched function by central differences.

    Every shifted point is evaluated in a single call of `function`.

    Args:
        function: Maps rows of shape `(N, P)` to values of shape `(N, ...)`.
        points: Points of shape `(N, P)` at which to differentiate.
        columns: Columns to differentiate with respect to. Defaults to all.
        step: Half the distance between the two evaluations of a column.

    Returns:
        Derivatives of shape `(N, len(columns), ...)`.
    """
    n_points, n_columns = points.shape
    chosen = list(range(n_columns)) if columns is None else list(columns)
    shifts = torch.zeros(len(chosen), n_columns, dtype=points.dtype)
    shifts[torch.arange(len(chosen)), chosen] = step

    plus = points[:, None, :] + shifts[None]
    minus = points[:, None, :] - shifts[None]
    rows = torch.cat([plus, minus]).reshape(-1, n_columns)

    values = function(rows)
    values = values.reshape(2, n_points, len(chosen), *values.shape[1:])
    return (values[0] - values[1]) / (2 * step)


def _costs(expectations: Tensor) -> Tensor:
    """Computes the global and the local cost from the expectation values.

    Args:
        expectations: `<Z_i>` of shape `(N, n_qubits)`.

    Returns:
        Shape `(N, 2)`: the mean of all `<Z_i>`, then `<Z_0>`.
    """
    return torch.stack([expectations.mean(dim=1), expectations[:, 0]], dim=1)


def _root_mean_square(values: Tensor) -> float:
    """Returns the root mean square of every entry."""
    return float(values.square().mean().sqrt())


def diagnose(
    choice: CircuitChoice,
    n_qubits: int,
    n_layers: int = DEFAULT_LAYERS,
    n_weight_draws: int = DEFAULT_WEIGHT_DRAWS,
    n_inputs: int = DEFAULT_INPUTS,
    seed: int = 0,
    step: float = DEFAULT_STEP,
    estimator: BaseEstimatorV2 | None = None,
) -> TrainabilityResult:
    """Measures one circuit at initialisation.

    Args:
        choice: The embedding, block, and re-uploading of the circuit.
        n_qubits: Number of qubits.
        n_layers: Number of block layers. Defaults to 4.
        n_weight_draws: Number of random weight initialisations.
        n_inputs: Number of random inputs per weight draw for the input spread.
        seed: Seed of every random draw.
        step: Central-difference step.
        estimator: Estimator for the exact simulation. If None, the default
            (Aer's noiseless state-vector estimator).

    Returns:
        The diagnostic of the circuit.
    """
    circuit, input_params, weight_vectors, observables = build_qiskit_ansatz(
        n_qubits=n_qubits,
        n_layers=n_layers,
        embedding=choice.embedding,
        block=choice.block,
        reupload=choice.reupload,
    )
    weight_params = [param for vector in weight_vectors for param in vector]
    runner = QiskitExpectationRunner(
        circuit, observables, (*input_params, *weight_params), estimator=estimator
    )
    embedding = get_embedding(choice.embedding)
    n_weights = len(weight_params)

    def expectations(rows: Tensor) -> Tensor:
        """Evaluates `<Z_i>` for rows of scaled features followed by weights."""
        angles = embedding.angles(rows[:, :n_qubits])
        return runner.expectation_values(torch.cat([angles, rows[:, n_qubits:]], 1))

    generator = torch.Generator().manual_seed(seed)
    weights = (
        torch.rand(n_weight_draws, n_weights, generator=generator, dtype=torch.float64)
        * 2
        - 1
    ) * math.pi
    features = (
        torch.rand(
            n_weight_draws,
            n_inputs,
            n_qubits,
            generator=generator,
            dtype=torch.float64,
        )
        - 0.5
    ) * math.pi

    spread_rows = torch.cat(
        [
            features.reshape(-1, n_qubits),
            weights.repeat_interleave(n_inputs, dim=0),
        ],
        dim=1,
    )
    outputs = expectations(spread_rows).reshape(n_weight_draws, n_inputs, n_qubits)
    input_spread = float(outputs.std(dim=1).min(dim=1).values.mean())

    def cost(rows: Tensor) -> Tensor:
        """Evaluates the global and the local cost for each row."""
        return _costs(expectations(rows))

    points = torch.cat([features[:, 0, :], weights], dim=1)
    input_gradients = central_differences(cost, points, range(n_qubits), step)
    weight_gradients = central_differences(
        cost, points, range(n_qubits, n_qubits + n_weights), step
    )
    variances = weight_gradients.var(dim=0).mean(dim=0)

    return TrainabilityResult(
        embedding=choice.embedding,
        block=choice.block,
        reupload=choice.reupload,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_input_angles=len(input_params),
        n_weights=n_weights,
        spsa_dimension=len(input_params) + n_weights,
        input_spread=input_spread,
        input_sensitivity_global=_root_mean_square(input_gradients[..., 0]),
        input_sensitivity_local=_root_mean_square(input_gradients[..., 1]),
        weight_gradient_rms_global=_root_mean_square(weight_gradients[..., 0]),
        weight_gradient_rms_local=_root_mean_square(weight_gradients[..., 1]),
        weight_gradient_variance_global=float(variances[0]),
        weight_gradient_variance_local=float(variances[1]),
    )


def default_choices() -> list[CircuitChoice]:
    """Lists the circuits the diagnostic covers by default.

    Every registered block with the paper's embedding, with and without
    re-uploading, then every other embedding with the paper's block (which
    repeats the earlier embedding comparison).

    Returns:
        The circuits, each once.
    """
    choices = [
        CircuitChoice(DEFAULT_EMBEDDING, block, reupload)
        for block in sorted(BLOCKS, key=lambda name: (name != DEFAULT_BLOCK, name))
        for reupload in (False, True)
    ]
    choices += [
        CircuitChoice(embedding, DEFAULT_BLOCK, False)
        for embedding in sorted(EMBEDDINGS)
        if embedding != DEFAULT_EMBEDDING
    ]
    return choices


def markdown_table(results: Sequence[TrainabilityResult]) -> str:
    """Renders results as a Markdown table, one row per result.

    Args:
        results: The diagnostics to show.

    Returns:
        The table, header and separator included.
    """
    header = (
        "| Embedding | Block | Re-upload | Qubits | Weights | SPSA dim "
        "| Input spread | Sensitivity, mean Z | Sensitivity, Z0 "
        "| Grad rms, mean Z | Grad var, mean Z | Grad var, Z0 |"
    )
    separator = "|" + " :-- |" * 12
    rows = [
        f"| {r.embedding} | {r.block} | {'yes' if r.reupload else 'no'} "
        f"| {r.n_qubits} | {r.n_weights} | {r.spsa_dimension} "
        f"| {r.input_spread:.3g} | {r.input_sensitivity_global:.3g} "
        f"| {r.input_sensitivity_local:.3g} | {r.weight_gradient_rms_global:.3g} "
        f"| {r.weight_gradient_variance_global:.3g} "
        f"| {r.weight_gradient_variance_local:.3g} |"
        for r in results
    ]
    return "\n".join([header, separator, *rows])
