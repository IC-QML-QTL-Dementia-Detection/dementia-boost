"""Expectation-value runner for Qiskit circuits on an injectable V2 estimator.

Owns a single responsibility: given a circuit, its Pauli observables, and a
batch of parameter values, return the expectation values. Which
`BaseEstimatorV2` implementation executes the circuit is a dependency chosen by
the caller, mirroring how `resolve_quantum_device` selects the PennyLane
device. Gradient estimation stays with the layer that uses this runner.
"""

from collections.abc import Sequence

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, BindingsArray
from qiskit.primitives.containers.bindings_array import ParameterLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from torch import Tensor

# Controlled rotations that qiskit-aer 0.17.2 evaluates wrongly when their angle
# is bound at run time (the gate behaves as the identity). The same circuit bound
# beforehand, or with these gates decomposed, gives the exact result, so the
# runner decomposes them once at construction.
_UNSAFE_AER_GATES: tuple[str, ...] = ("cry", "crz", "cp")


def resolve_qiskit_estimator(
    estimator: BaseEstimatorV2 | None = None,
) -> BaseEstimatorV2:
    """Resolves the Qiskit Primitives V2 estimator that runs the circuits.

    The default is Aer's `EstimatorV2` on its state-vector method with no noise
    model attached, so it is the fast noiseless simulator, not a noisy backend.
    Qiskit's `StatevectorEstimator` is the slower reference implementation and
    should be injected explicitly where its role as a reference is wanted.

    Args:
        estimator: Optional pre-built estimator, returned unchanged. If None,
            resolves to a noiseless Aer state-vector `EstimatorV2`.

    Returns:
        The estimator to execute circuits with.
    """
    if estimator is not None:
        return estimator

    return AerEstimator(options={"backend_options": {"method": "statevector"}})


class QiskitExpectationRunner:
    """Evaluates all Pauli expectation values of a circuit in one broadcast PUB.

    Each call submits one PUB whose parameter batch has shape `(N, 1, P)` and
    whose observables have shape `(1, n_observables)`, so every one of the `N`
    states is evolved once and all observables are read from it.

    Attributes:
        estimator: The Primitives V2 estimator executing the circuit.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: Sequence[SparsePauliOp],
        parameters: Sequence[ParameterLike],
        estimator: BaseEstimatorV2 | None = None,
    ) -> None:
        """Binds the circuit, observables, parameter order, and estimator.

        Controlled rotations that Aer mishandles with run-time bound angles are
        decomposed first (see `_UNSAFE_AER_GATES`).

        Args:
            circuit: The parameterized circuit to evaluate.
            observables: Observables whose expectation values are returned,
                one output column each.
            parameters: The circuit parameters in the column order of the
                values later passed to `expectation_values`.
            estimator: Optional pre-built estimator. If None, resolves through
                `resolve_qiskit_estimator`.
        """
        self.estimator = resolve_qiskit_estimator(estimator)
        self._circuit = circuit.decompose(gates_to_decompose=list(_UNSAFE_AER_GATES))
        self._observables = ObservablesArray([list(observables)])
        self._parameters = tuple(parameters)

    def expectation_values(self, params: Tensor) -> Tensor:
        """Evaluates every observable for every row of parameter values.

        Args:
            params: Parameter values of shape `(N, P)`, with columns ordered
                like the `parameters` given at construction.

        Returns:
            Tensor of shape `(N, n_observables)` with the exact expectation
            values, on the device of `params`.
        """
        values = params.detach().cpu().numpy().astype(np.float64)[:, None, :]

        pub = EstimatorPub(
            self._circuit,
            self._observables,
            BindingsArray({self._parameters: values}),
            precision=0.0,
        )
        result = self.estimator.run([pub]).result()[0]

        return torch.from_numpy(np.asarray(result.data["evs"])).to(params.device)
