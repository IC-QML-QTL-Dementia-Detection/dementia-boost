"""Tests of the per-group training instrumentation.

The monitor measures, per optimizer group, how far the weights moved from
their initial values and how large the gradients were. These tests pin the
definitions, so the numbers in the histories keep one meaning: the
displacement is measured from the weights at the start of training (not from
the previous epoch) and the gradient norm is the mean over the recorded steps.
"""

import math

import pytest
import torch
import torch.nn as nn

from dementia_boost.training.instrumentation import ParameterGroupMonitor
from dementia_boost.training.optimization import MODEL_GROUP


def test_monitor_measures_displacement_from_the_start_and_mean_gradient() -> None:
    """Before any step the displacement is 0; after the weights move it is
    the RMS and maximum of the change since the start, and the gradient norm
    is the mean L2 norm of the recorded steps, reset after each summary."""
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    monitor = ParameterGroupMonitor(optimizer)

    (start,) = monitor.summarize([0.1]).values()
    assert (start.displacement_rms, start.displacement_max) == (0.0, 0.0)
    assert start.grad_norm_mean == 0.0

    weight.grad = torch.tensor([3.0, 4.0])
    monitor.record_gradients()
    weight.grad = torch.tensor([0.0, 1.0])
    monitor.record_gradients()
    with torch.no_grad():
        weight += torch.tensor([0.3, -0.4])
    first = monitor.summarize([0.1])

    assert list(first) == [MODEL_GROUP]
    assert first[MODEL_GROUP].lr == 0.1
    assert first[MODEL_GROUP].grad_norm_mean == pytest.approx(3.0)
    assert first[MODEL_GROUP].displacement_max == pytest.approx(0.4)
    assert first[MODEL_GROUP].displacement_rms == pytest.approx(
        math.sqrt((0.09 + 0.16) / 2)
    )

    with torch.no_grad():
        weight += torch.tensor([0.0, 0.4])
    second = monitor.summarize([0.05])[MODEL_GROUP]

    assert second.displacement_max == pytest.approx(0.3)
    assert second.grad_norm_mean == 0.0
    assert second.lr == 0.05
