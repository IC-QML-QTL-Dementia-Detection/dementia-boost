"""Tests of the optimizer factory: named parameter groups and their rates.

The quantum head trains its circuit weights in a group of their own, so the
circuit can get a learning rate (`quantum_lr`) apart from the classical layers.
These tests guard that the groups split the parameters exactly, that each group
gets the rate the spec names, that leaving `quantum_lr` unset trains exactly as
one shared group did (so existing configuration IDs still mean the same
training), and that one `StepLR` decays every group from its own base rate.
"""

import copy

import pytest
import torch
import torch.nn as nn
from conftest import build_spec

from dementia_boost.core.identity import Paradigm, RunSpec
from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.training.optimization import (
    CIRCUIT_GROUP,
    MODEL_GROUP,
    build_optimizer,
    build_scheduler,
)

IN_FEATURES: int = 8
LEARNING_RATE: float = 1e-3
QUANTUM_LEARNING_RATE: float = 5e-2


def _qtl_head() -> QiskitQuantumClassifierHead:
    """A small quantum head (2 qubits, 1 layer) on 8 input features."""
    torch.manual_seed(0)
    return QiskitQuantumClassifierHead(
        in_features=IN_FEATURES, n_qubits=2, n_layers=1, seed=0
    )


def _qtl_spec(**overrides) -> RunSpec:
    """A Qiskit QTL spec matching `_qtl_head`."""
    return build_spec(Paradigm.QISKIT_QTL, lr=LEARNING_RATE, **overrides)


def _rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Maps each group name of an optimizer to its learning rate."""
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


def test_quantum_head_groups_are_disjoint_and_cover_every_parameter() -> None:
    """Every trainable parameter is in exactly one named group, so no weight
    is trained twice or left out."""
    head = _qtl_head()

    optimizer = build_optimizer(head, _qtl_spec(quantum_lr=QUANTUM_LEARNING_RATE))

    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in head.parameters())
    assert len(grouped) == len(set(grouped))
    assert [group["name"] for group in optimizer.param_groups] == [
        "pre_net",
        CIRCUIT_GROUP,
        "post_net",
    ]
    (circuit,) = [g for g in optimizer.param_groups if g["name"] == CIRCUIT_GROUP]
    assert circuit["params"] == [head.qnn.weight]


def test_groups_carry_the_rates_of_the_spec() -> None:
    """The circuit group gets `quantum_lr`, the classical groups `lr`; without
    `quantum_lr` the circuit shares `lr`."""
    separate = build_optimizer(_qtl_head(), _qtl_spec(quantum_lr=QUANTUM_LEARNING_RATE))
    shared = build_optimizer(_qtl_head(), _qtl_spec())

    assert _rates(separate) == {
        "pre_net": LEARNING_RATE,
        CIRCUIT_GROUP: QUANTUM_LEARNING_RATE,
        "post_net": LEARNING_RATE,
    }
    assert set(_rates(shared).values()) == {LEARNING_RATE}


def test_a_model_without_groups_is_one_group_at_lr() -> None:
    """A classical model has no named groups and trains as one group."""
    model = nn.Linear(IN_FEATURES, 1)

    optimizer = build_optimizer(model, build_spec(Paradigm.CTL, lr=LEARNING_RATE))

    assert _rates(optimizer) == {MODEL_GROUP: LEARNING_RATE}
    assert optimizer.param_groups[0]["params"] == list(model.parameters())


def test_shared_rate_steps_exactly_like_a_single_group() -> None:
    """With `quantum_lr` unset, the grouped optimizer moves every weight
    exactly as the single-group Adam used before the groups existed, so the
    configurations trained before keep their meaning."""
    grouped_head = _qtl_head()
    single_head = copy.deepcopy(grouped_head)
    grouped = build_optimizer(grouped_head, _qtl_spec())
    single = torch.optim.Adam(single_head.parameters(), lr=LEARNING_RATE)

    generator = torch.Generator().manual_seed(1)
    for _ in range(3):
        for grouped_param, single_param in zip(
            grouped_head.parameters(), single_head.parameters(), strict=True
        ):
            gradient = torch.randn(grouped_param.shape, generator=generator)
            grouped_param.grad = gradient.clone()
            single_param.grad = gradient.clone()
        grouped.step()
        single.step()

    for grouped_param, single_param in zip(
        grouped_head.parameters(), single_head.parameters(), strict=True
    ):
        assert torch.equal(grouped_param, single_param)


def test_step_lr_decays_every_group_from_its_own_base_rate() -> None:
    """One `StepLR` scales each group by the same factor, so the circuit keeps
    its own rate relative to the classical layers through the schedule."""
    spec = _qtl_spec(quantum_lr=QUANTUM_LEARNING_RATE, lr_step_size=2, lr_gamma=0.5)
    optimizer = build_optimizer(_qtl_head(), spec)
    scheduler = build_scheduler(optimizer, spec)

    for _ in range(2 * spec.lr_step_size):
        optimizer.step()
        scheduler.step()

    assert _rates(optimizer) == pytest.approx(
        {
            "pre_net": LEARNING_RATE * 0.25,
            CIRCUIT_GROUP: QUANTUM_LEARNING_RATE * 0.25,
            "post_net": LEARNING_RATE * 0.25,
        }
    )
