"""Optimizer and learning-rate schedule built from a run specification.

The optimizer is Adam with named parameter groups. A model that names its
groups (the quantum head: `pre_net`, `circuit`, `post_net`) gets one group per
name; any other model is a single group, `model`. The circuit group trains at
`spec.quantum_lr`, or at `spec.lr` when that field is unset; every other group
trains at `spec.lr`. Adam keeps its state per parameter, so with the shared rate
the groups train exactly as one group would.

One `StepLR` decays every group by the same factor, each from its own base rate.
"""

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.identity import RunSpec

CIRCUIT_GROUP: str = "circuit"
MODEL_GROUP: str = "model"


@runtime_checkable
class NamedParameterGroups(Protocol):
    """A model that splits its trainable parameters into named groups."""

    def parameter_groups(self) -> Mapping[str, Iterable[nn.Parameter]]:
        """Returns the trainable parameters by group name, in a fixed order."""
        ...


def group_learning_rate(name: str, spec: RunSpec) -> float:
    """Returns the learning rate the spec gives a parameter group.

    Args:
        name: Name of the parameter group.
        spec: The run specification.

    Returns:
        `spec.quantum_lr` for the circuit group when it is set, otherwise
        `spec.lr`.
    """
    if name == CIRCUIT_GROUP and spec.quantum_lr is not None:
        return spec.quantum_lr
    return spec.lr


def build_optimizer(model: nn.Module, spec: RunSpec) -> Optimizer:
    """Builds Adam over the model's named parameter groups.

    Args:
        model: The module to train. If it implements `parameter_groups`, each
            group becomes an optimizer group under its name; otherwise all its
            parameters form the single group `MODEL_GROUP`.
        spec: The run specification giving the learning rates.

    Returns:
        Adam with one group per name, each carrying a `name` entry and the
        learning rate `group_learning_rate` gives it.
    """
    if isinstance(model, NamedParameterGroups):
        named = model.parameter_groups()
    else:
        named = {MODEL_GROUP: model.parameters()}

    return Adam(
        [
            {
                "params": list(params),
                "lr": group_learning_rate(name, spec),
                "name": name,
            }
            for name, params in named.items()
        ]
    )


def build_scheduler(optimizer: Optimizer, spec: RunSpec) -> StepLR:
    """Builds the `StepLR` schedule the spec describes, shared by every group.

    Args:
        optimizer: The optimizer whose groups are decayed.
        spec: The run specification giving `lr_step_size` and `lr_gamma`.

    Returns:
        A `StepLR` that scales every group by `lr_gamma` every `lr_step_size`
        epochs.
    """
    return StepLR(optimizer, step_size=spec.lr_step_size, gamma=spec.lr_gamma)
