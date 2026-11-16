"""Acceptance tests run on every registered circuit configuration.

Every combination of a registered embedding, a registered variational block,
and re-uploading on or off must pass the same guards, so a new registry entry
is tested without new test code:

- each `<Z_i>` varies with the inputs at fixed weights;
- the output varies with the circuit weights;
- the SPSA backward pass gives clearly non-zero gradients for the scaled
  inputs and for the circuit weights;
- the default Aer estimator agrees with Qiskit's `StatevectorEstimator`.

Regression coverage
-------------------
- The circuit degenerating into a constant function (every `<Z_i>` equal to 1,
  zero gradients), as the paper's gate sequence does without a Hadamard layer.
- qiskit-aer evaluating controlled rotations with run-time bound angles as the
  identity, which the runner works around for the gates it knows; a gate in a
  new block that Aer mishandles fails the parity guard.
"""

import itertools
import math
from functools import partial

import pytest
import torch
from qiskit.primitives import StatevectorEstimator

from dementia_boost.models.quantum_cnn import (
    BLOCKS,
    EMBEDDINGS,
    QiskitExpectationRunner,
    build_qiskit_ansatz,
    get_embedding,
)
from dementia_boost.models.quantum_cnn.qiskit_layer import (
    QiskitQuantumLayer,
    create_qiskit_quantum_layer,
)

_N_QUBITS: int = 3
_N_LAYERS: int = 2
_BATCH: int = 32
_MIN_SPREAD: float = 1e-3
_MIN_GRADIENT: float = 1e-6
_PARITY_TOLERANCE: float = 1e-10

_CONFIGURATIONS = [
    pytest.param(
        embedding,
        block,
        reupload,
        id=f"{embedding}-{block}-{'reupload' if reupload else 'once'}",
    )
    for embedding, block, reupload in itertools.product(
        sorted(EMBEDDINGS), sorted(BLOCKS), (False, True)
    )
]


def _layer(
    embedding: str, block: str, reupload: bool, seed: int = 0
) -> QiskitQuantumLayer:
    """Creates a seeded layer for one configuration on the default estimator."""
    torch.manual_seed(seed)
    return create_qiskit_quantum_layer(
        n_qubits=_N_QUBITS,
        n_layers=_N_LAYERS,
        seed=seed,
        ansatz_builder=partial(
            build_qiskit_ansatz, embedding=embedding, block=block, reupload=reupload
        ),
    )


def _scaled_inputs(seed: int) -> torch.Tensor:
    """Draws scaled features in `[-pi/2, pi/2]`, the range the head produces."""
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(_BATCH, _N_QUBITS, generator=generator) - 0.5) * math.pi


@pytest.mark.parametrize(("embedding", "block", "reupload"), _CONFIGURATIONS)
class TestEveryConfiguration:
    """Guards that every embedding, block, and re-uploading choice must pass."""

    def test_every_expectation_value_varies_with_the_inputs(
        self, embedding: str, block: str, reupload: bool
    ) -> None:
        """Asserts each qubit's `<Z_i>` spreads across random inputs."""
        layer = _layer(embedding, block, reupload)
        angles = get_embedding(embedding).angles(_scaled_inputs(seed=1))

        with torch.no_grad():
            outputs = layer(angles)

        assert torch.all(outputs.std(dim=0) > _MIN_SPREAD)

    def test_output_varies_with_the_weights(
        self, embedding: str, block: str, reupload: bool
    ) -> None:
        """Asserts two weight draws give clearly different outputs."""
        angles = get_embedding(embedding).angles(_scaled_inputs(seed=1))

        with torch.no_grad():
            first = _layer(embedding, block, reupload, seed=0)(angles)
            second = _layer(embedding, block, reupload, seed=1)(angles)

        assert (first - second).abs().max() > _MIN_SPREAD

    def test_spsa_gradients_reach_the_inputs_and_the_weights(
        self, embedding: str, block: str, reupload: bool
    ) -> None:
        """Asserts the backward pass gives clearly non-zero gradients for the
        scaled inputs (through the embedding's angle function) and for the
        circuit weights."""
        layer = _layer(embedding, block, reupload)
        inputs = _scaled_inputs(seed=1).requires_grad_()

        output = layer(get_embedding(embedding).angles(inputs))
        upstream = torch.randn(output.shape, generator=torch.Generator().manual_seed(2))
        (output * upstream).sum().backward()

        assert inputs.grad is not None
        assert layer.weight.grad is not None
        assert inputs.grad.abs().max() > _MIN_GRADIENT
        assert layer.weight.grad.abs().max() > _MIN_GRADIENT

    def test_default_estimator_matches_the_reference(
        self, embedding: str, block: str, reupload: bool
    ) -> None:
        """Asserts Aer and `StatevectorEstimator` give the same `<Z_i>` on the
        same random parameters."""
        circuit, input_params, weight_vectors, observables = build_qiskit_ansatz(
            n_qubits=_N_QUBITS,
            n_layers=_N_LAYERS,
            embedding=embedding,
            block=block,
            reupload=reupload,
        )
        parameters = (*input_params, *(p for v in weight_vectors for p in v))
        default = QiskitExpectationRunner(circuit, observables, parameters)
        reference = QiskitExpectationRunner(
            circuit, observables, parameters, estimator=StatevectorEstimator()
        )

        generator = torch.Generator().manual_seed(3)
        values = (
            torch.rand(8, len(parameters), generator=generator, dtype=torch.float64)
            * 2
            * math.pi
        )

        assert torch.allclose(
            default.expectation_values(values),
            reference.expectation_values(values),
            atol=_PARITY_TOLERANCE,
        )
