"""Unit tests for the trainability diagnostic at initialisation.

The diagnostic is a measurement, not a guard: these tests check that its
derivatives are right, that it is reproducible from its seed, and that it
reports every quantity, not what the values should be for a given circuit.
"""

import math

import torch

from dementia_boost.analysis.trainability import (
    CircuitChoice,
    TrainabilityResult,
    central_differences,
    default_choices,
    diagnose,
)
from dementia_boost.models.quantum_cnn import BLOCKS, EMBEDDINGS


def _diagnose_small(choice: CircuitChoice, seed: int) -> TrainabilityResult:
    """Runs the diagnostic on two qubits and one layer, with few draws."""
    return diagnose(
        choice, n_qubits=2, n_layers=1, n_weight_draws=3, n_inputs=4, seed=seed
    )


def test_central_differences_match_the_derivative_of_the_cosine() -> None:
    """The derivative of `cos(x)` is `-sin(x)`, column by column."""
    points = torch.tensor([[0.3, -1.2], [2.0, 0.5]], dtype=torch.float64)

    gradients = central_differences(lambda values: torch.cos(values).sum(1), points)

    assert torch.allclose(gradients, -torch.sin(points), atol=1e-8)


def test_the_same_seed_gives_the_same_result_and_another_seed_does_not() -> None:
    """Random weights and inputs come only from the given seed."""
    choice = CircuitChoice(embedding="paper", block="paper", reupload=False)

    first = _diagnose_small(choice, seed=0)
    again = _diagnose_small(choice, seed=0)
    other = _diagnose_small(choice, seed=1)

    assert first == again
    assert first != other


def test_reports_counts_and_finite_statistics() -> None:
    """The paper's block on two qubits and one layer has 2 input angles and 6
    weights, so SPSA perturbs 8 values; every statistic is finite."""
    result = _diagnose_small(CircuitChoice("paper", "paper", False), seed=0)

    assert (result.n_input_angles, result.n_weights, result.spsa_dimension) == (2, 6, 8)
    statistics = [
        result.input_spread,
        result.input_sensitivity_global,
        result.input_sensitivity_local,
        result.weight_gradient_rms_global,
        result.weight_gradient_rms_local,
        result.weight_gradient_variance_global,
        result.weight_gradient_variance_local,
    ]
    assert all(math.isfinite(value) and value >= 0 for value in statistics)


def test_default_choices_cover_every_block_and_every_embedding() -> None:
    """Every block appears with the paper's embedding, with and without
    re-uploading, and every embedding with the paper's block, each once."""
    choices = default_choices()

    assert len(choices) == len(set(choices))
    for block in BLOCKS:
        assert CircuitChoice("paper", block, False) in choices
        assert CircuitChoice("paper", block, True) in choices
    for embedding in EMBEDDINGS:
        assert CircuitChoice(embedding, "paper", False) in choices
