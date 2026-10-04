"""PyTorch bridge for Qiskit circuits built directly on Qiskit Primitives V2.

Evaluates the Qiskit ansatz with a `StatevectorEstimator` and exposes it as a
PyTorch `nn.Module` through a custom `torch.autograd.Function`. Each forward
pass is a single broadcast PUB, so every sample is simulated once and all
Pauli-Z expectation values are read from that one state. The backward pass is
a loss-level SPSA estimate that also costs a single broadcast PUB.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import BindingsArray, StatevectorEstimator
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .qiskit_circuit import DEFAULT_N_LAYERS, DEFAULT_N_QUBITS, build_qiskit_ansatz

DEFAULT_SPSA_EPSILON: float = 0.01

AnsatzParts = tuple[
    QuantumCircuit, ParameterVector, list[ParameterVector], list[SparsePauliOp]
]
AnsatzBuilder = Callable[..., AnsatzParts]


class _ExpectationContext(FunctionCtx):
    """Typed autograd context carrying what the SPSA backward pass needs.

    Attributes:
        saved_tensors: Tensors stashed with `save_for_backward`.
        layer: The layer that owns the estimator and SPSA settings.
        n_inputs: Number of input-angle columns in the flat parameter tensor.
    """

    saved_tensors: tuple[Tensor, ...]
    layer: "QiskitQuantumLayer"
    n_inputs: int


class _EstimatorExpectation(torch.autograd.Function):
    """Autograd function for the circuit's Pauli-Z expectation values.

    The backward pass is a simultaneous perturbation (SPSA) estimate taken at
    the loss level: one Rademacher direction per sample is shared by all
    outputs by folding the upstream gradient into the finite difference.
    """

    @staticmethod
    def forward(
        ctx: _ExpectationContext,
        angles: Tensor,
        weight: Tensor,
        layer: "QiskitQuantumLayer",
    ) -> Tensor:
        """Evaluates `<Z_i>` for every sample with one broadcast PUB.

        Args:
            ctx: Autograd context used to stash tensors for the backward pass.
            angles: Embedded input angles of shape `(Batch, n_qubits)`.
            weight: Flat circuit weight vector of shape `(n_weights,)`.
            layer: The owning layer, providing the estimator and SPSA settings.

        Returns:
            Expectation values of shape `(Batch, n_qubits)`, in `angles.dtype`.
        """
        params = torch.cat([angles, weight.expand(angles.shape[0], -1)], dim=1)
        ctx.save_for_backward(params)
        ctx.layer = layer
        ctx.n_inputs = angles.shape[1]

        return layer.expectation_values(params).to(angles.dtype)

    @staticmethod
    def backward(
        ctx: _ExpectationContext, *grad_outputs: Tensor
    ) -> tuple[Tensor, Tensor, None]:
        """Estimates the input and weight gradients with one SPSA direction per sample.

        With a Rademacher direction `delta` and upstream vector `v`, the
        directional derivative of the loss is
        `v^T (E(p + eps * delta) - E(p - eps * delta)) / (2 * eps)`, and the
        gradient estimate is that scalar times `delta`. Both shifted batches
        are evaluated in a single PUB of `2 * Batch` states.

        Args:
            ctx: Autograd context holding the forward parameters.
            *grad_outputs: The single upstream gradient `dL/d<Z_i>` of shape
                `(Batch, n_qubits)`.

        Returns:
            A tuple with the gradients for `angles`, `weight`, and `None` for
            the non-tensor `layer` argument.
        """
        (grad_output,) = grad_outputs
        (params,) = ctx.saved_tensors
        layer = ctx.layer

        delta = layer.sample_directions(params)
        epsilon = layer.spsa_epsilon
        shifted = torch.cat([params + epsilon * delta, params - epsilon * delta])
        plus, minus = layer.expectation_values(shifted).to(params.dtype).chunk(2)

        directional = ((plus - minus) * grad_output).sum(dim=1) / (2 * epsilon)
        gradient = directional[:, None] * delta

        return gradient[:, : ctx.n_inputs], gradient[:, ctx.n_inputs :].sum(dim=0), None


class QiskitQuantumLayer(nn.Module):
    """Variational Qiskit circuit exposed as a differentiable PyTorch layer.

    Holds the circuit weights, uniformly initialized over `[-pi, pi]`, in one
    flat `weight` parameter laid out as `[theta, gamma, beta]`. Expectation
    values come from a noiseless `StatevectorEstimator` (precision 0.0), and
    gradients from SPSA with perturbations drawn from a dedicated generator.

    Attributes:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of ansatz layer repetitions.
        spsa_epsilon: Finite-difference step of the SPSA gradient estimate.
        weight: Flat trainable circuit weights of shape `(3 * n_layers * n_qubits,)`.
    """

    def __init__(
        self,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        spsa_epsilon: float = DEFAULT_SPSA_EPSILON,
        seed: int | None = None,
        ansatz_builder: AnsatzBuilder = build_qiskit_ansatz,
    ) -> None:
        """Builds the circuit, observables, estimator, and trainable weights.

        Args:
            n_qubits: Number of qubits in the circuit. Defaults to 6.
            n_layers: Number of ansatz layer repetitions. Defaults to 4.
            spsa_epsilon: Finite-difference step of the SPSA gradient estimate.
                Defaults to `DEFAULT_SPSA_EPSILON`.
            seed: Seed of the generator that draws the SPSA perturbation
                signs. If None, uses `torch.initial_seed()`, which is the value
                `core.reproducibility.set_seed` locked for the current run.
            ansatz_builder: Callable with the signature and return value of
                `build_qiskit_ansatz`. Defaults to the Bhowmik et al. (2025)
                ansatz, and lets other circuits share this layer.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.spsa_epsilon = spsa_epsilon

        self._circuit, input_params, weight_vectors, observables = ansatz_builder(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )
        weight_params = [param for vector in weight_vectors for param in vector]

        self._flat_parameters = (*input_params, *weight_params)
        self._observables = ObservablesArray([observables])
        self._estimator = StatevectorEstimator()

        self._generator = torch.Generator()
        self._generator.manual_seed(seed if seed is not None else torch.initial_seed())

        self.weight = nn.Parameter(
            torch.rand(len(weight_params), dtype=torch.float32) * 2 * torch.pi
            - torch.pi
        )

    def expectation_values(self, params: Tensor) -> Tensor:
        """Evaluates all Pauli-Z expectation values in one broadcast PUB.

        Args:
            params: Flat circuit parameters of shape `(N, n_qubits + n_weights)`,
                inputs first and weights after, matching the layer's layout.

        Returns:
            Tensor of shape `(N, n_qubits)` with the exact `<Z_i>` values.
        """
        values = params.detach().cpu().numpy().astype(np.float64)[:, None, :]

        pub = EstimatorPub(
            self._circuit,
            self._observables,
            BindingsArray({self._flat_parameters: values}),
            precision=0.0,
        )
        result = self._estimator.run([pub]).result()[0]

        return torch.from_numpy(np.asarray(result.data["evs"])).to(params.device)

    def sample_directions(self, params: Tensor) -> Tensor:
        """Draws one Rademacher perturbation direction per sample.

        Args:
            params: Flat circuit parameters of shape `(Batch, n_params)`.

        Returns:
            Tensor of `+1`/`-1` entries with the shape and dtype of `params`.
        """
        signs = torch.randint(
            0, 2, params.shape, generator=self._generator, device="cpu"
        )
        return (signs * 2 - 1).to(device=params.device, dtype=params.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the Pauli-Z expectation values for a batch of angles.

        Args:
            x: Embedded input angles of shape `(Batch, n_qubits)`.

        Returns:
            Expectation values of shape `(Batch, n_qubits)`, each in `[-1, 1]`.
        """
        return cast(Tensor, _EstimatorExpectation.apply(x, self.weight, self))


def create_qiskit_quantum_layer(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    spsa_epsilon: float = DEFAULT_SPSA_EPSILON,
    seed: int | None = None,
    ansatz_builder: AnsatzBuilder = build_qiskit_ansatz,
) -> QiskitQuantumLayer:
    """Instantiates the Qiskit variational layer for the Bhowmik et al. (2025) ansatz.

    Mirrors `create_quantum_layer` on the PennyLane side. Expectation values
    are exact and noiseless. Gradients flow to both the circuit weights and the
    upstream classical inputs, so autograd can backpropagate through the
    pre-net feeding this layer. They are SPSA estimates, because exact
    parameter-shift needs 204 shifted circuits per sample at the default
    6-qubit, 4-layer depth.

    Args:
        n_qubits: Number of qubits in the circuit. Defaults to 6.
        n_layers: Number of ansatz layer repetitions. Defaults to 4.
        spsa_epsilon: Finite-difference step of the SPSA gradient estimate.
        seed: Seed of the SPSA perturbation generator. If None, follows the
            seed locked by `core.reproducibility.set_seed`.
        ansatz_builder: Callable returning the circuit, input parameters,
            weight parameter vectors, and observables. Defaults to
            `build_qiskit_ansatz`.

    Returns:
        A `QiskitQuantumLayer` with weights uniformly initialized over
        `[-pi, pi]`.
    """
    return QiskitQuantumLayer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        spsa_epsilon=spsa_epsilon,
        seed=seed,
        ansatz_builder=ansatz_builder,
    )
