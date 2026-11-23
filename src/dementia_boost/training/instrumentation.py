"""Per-group measurements of how training moves the weights.

`ParameterGroupMonitor` follows the groups of an optimizer through training:
how far each group's weights have moved from where they started, and how large
the gradients the optimizer received were. For the quantum head this tells
whether the circuit weights leave their random initialisation at a given
learning rate; for a classical head, a group whose gradient stays at zero
shows a layer that stopped learning.
"""

from collections.abc import Sequence

import torch
from torch.optim import Optimizer

from dementia_boost.telemetry.metrics import GroupRecord
from dementia_boost.training.optimization import group_names


class ParameterGroupMonitor:
    """Measures displacement and gradient norm per optimizer group.

    The initial weights are copied on construction, so build the monitor
    right before the first optimization step.

    Attributes:
        names: Name of each optimizer group, in order (`MODEL_GROUP` for a
            group without a `name` entry).
    """

    def __init__(self, optimizer: Optimizer) -> None:
        """Copies the initial weights of every optimizer group.

        Args:
            optimizer: The optimizer whose groups are followed.
        """
        self.names = group_names(optimizer)
        self._params = [list(group["params"]) for group in optimizer.param_groups]
        self._initial = [
            torch.cat([p.detach().flatten() for p in params]).clone()
            for params in self._params
        ]
        self._grad_norm_sums = [0.0] * len(self._params)
        self._steps = 0

    def record_gradients(self) -> None:
        """Adds the current gradient norm of every group to the epoch's sum.

        Call once per optimization step, after the backward pass. A parameter
        without a gradient counts as a zero gradient.
        """
        for index, params in enumerate(self._params):
            squared = sum(
                float(p.grad.detach().pow(2).sum())
                for p in params
                if p.grad is not None
            )
            self._grad_norm_sums[index] += squared**0.5
        self._steps += 1

    def summarize(self, learning_rates: Sequence[float]) -> dict[str, GroupRecord]:
        """Returns every group's measurements and starts a new epoch.

        Args:
            learning_rates: Learning rate of each group during the epoch, in
                the optimizer's order.

        Returns:
            One `GroupRecord` per group name, in the optimizer's order: the
            displacement of the current weights from the initial ones, and
            the mean gradient norm over the steps recorded since the last
            summary (0 when none were).
        """
        records: dict[str, GroupRecord] = {}
        for name, params, initial, grad_sum, lr in zip(
            self.names,
            self._params,
            self._initial,
            self._grad_norm_sums,
            learning_rates,
            strict=True,
        ):
            current = torch.cat([p.detach().flatten() for p in params])
            displacement = current - initial
            records[name] = GroupRecord(
                lr=float(lr),
                displacement_rms=float(displacement.pow(2).mean().sqrt()),
                displacement_max=float(displacement.abs().max()),
                grad_norm_mean=grad_sum / self._steps if self._steps else 0.0,
            )

        self._grad_norm_sums = [0.0] * len(self._params)
        self._steps = 0
        return records
