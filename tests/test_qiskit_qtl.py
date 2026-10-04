"""Unit tests for the Qiskit v2.x Quantum Transfer Learning (QTL) pipeline.

This module validates:
- ``build_qiskit_ansatz`` circuit shape, parameter counts, and observable
  construction across dynamic qubit/layer configurations.
- ``create_qiskit_quantum_layer`` forward-pass agreement with an independent
  exact ``Statevector`` evaluation, direction of its SPSA backward pass
  against an exact finite-difference gradient, and seeded reproducibility of
  the SPSA perturbations.
- ``QiskitQuantumClassifierHead`` end-to-end forward/backward pass and
  Glorot init isolation from quantum circuit weights.

Regression coverage
--------------------
- Drift in the ansatz's theta/gamma/beta parameter counts breaking dynamic
  qubit/layer configuration.
- Parameter or qubit ordering mismatches between the layer's flat weight
  vector and the circuit, which would silently evaluate a different circuit.
- A biased or mis-scaled SPSA backward pass, including the loss-level
  vector-Jacobian folding and the gradient flowing into the classical pre-net.
- SPSA perturbations ignoring the run seed, breaking reproducibility.
- Classical re-initialization accidentally perturbing quantum circuit weights.
"""

import math
from collections.abc import Iterable

import torch
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import (
    PrimitiveJob,
    PrimitiveResult,
    PubResult,
    StatevectorEstimator,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPubLike
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from dementia_boost.models.quantum_cnn import (
    QiskitExpectationRunner,
    QiskitQuantumClassifierHead,
)
from dementia_boost.models.quantum_cnn.qiskit_circuit import build_qiskit_ansatz
from dementia_boost.models.quantum_cnn.qiskit_layer import (
    QiskitQuantumLayer,
    create_qiskit_quantum_layer,
)

_EXPECTATION_TOLERANCE: float = 1e-5
_PARITY_TOLERANCE: float = 1e-10
_FINITE_DIFFERENCE_STEP: float = 1e-3
_MIN_SPSA_COSINE_SIMILARITY: float = 0.9


def _hadamard_ansatz(
    n_qubits: int, n_layers: int
) -> tuple[QuantumCircuit, ParameterVector, list[ParameterVector], list[SparsePauliOp]]:
    """Builds the ansatz with a Hadamard layer in front, for tests only.

    The paper ansatz keeps the state at `|0...0>`, so its expectation values
    and gradients are identically constant and cannot exercise the layer. The
    Hadamard layer leaves the gate sequence, the parameters, and the
    observables untouched, but makes outputs depend on both inputs and
    weights.

    Args:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of ansatz layer repetitions.

    Returns:
        The same tuple as `build_qiskit_ansatz`.
    """
    circuit, input_params, weight_vectors, observables = build_qiskit_ansatz(
        n_qubits=n_qubits, n_layers=n_layers
    )
    prepared = QuantumCircuit(n_qubits)
    prepared.h(range(n_qubits))
    prepared.compose(circuit, inplace=True)

    return prepared, input_params, weight_vectors, observables


def _hadamard_layer(n_qubits: int, n_layers: int, seed: int) -> QiskitQuantumLayer:
    """Creates a seeded layer around the test-only Hadamard ansatz."""
    return create_qiskit_quantum_layer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
        ansatz_builder=_hadamard_ansatz,
        estimator=StatevectorEstimator(),
    )


def _exact_expectations(
    layer: QiskitQuantumLayer, inputs: torch.Tensor
) -> torch.Tensor:
    """Evaluates the Hadamard-ansatz layer exactly with `Statevector`, per row.

    Args:
        layer: A layer built from `_hadamard_ansatz`, supplying the flat weights.
        inputs: Angle tensor of shape `(Batch, n_qubits)`.

    Returns:
        Tensor of shape `(Batch, n_qubits)` with the exact `<Z_i>` values.
    """
    circuit, input_params, weight_vectors, observables = _hadamard_ansatz(
        n_qubits=layer.n_qubits, n_layers=layer.n_layers
    )
    weight_params = [param for vector in weight_vectors for param in vector]

    rows = []
    for row in inputs:
        bound: QuantumCircuit = circuit.assign_parameters(
            {
                **dict(zip(input_params, row.tolist(), strict=True)),
                **dict(zip(weight_params, layer.weight.tolist(), strict=True)),
            }
        )
        state = Statevector(bound)
        rows.append([state.expectation_value(op).real for op in observables])

    return torch.tensor(rows, dtype=inputs.dtype)


def _finite_difference_gradients(
    layer: QiskitQuantumLayer,
    inputs: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the exact gradient of `sum(upstream * layer(inputs))`.

    Central differences on the noiseless, smooth expectation values give an
    exact reference at this step size, unlike the stochastic SPSA estimate.

    Args:
        layer: The layer to differentiate. Its weights are restored on exit.
        inputs: Angle tensor of shape `(Batch, n_qubits)`.
        upstream: Weights `v` of shape `(Batch, n_qubits)` applied to the outputs.

    Returns:
        A tuple `(input_grad, weight_grad)` shaped like `inputs` and `layer.weight`.
    """

    def scalar(x: torch.Tensor) -> float:
        with torch.no_grad():
            return float((layer(x) * upstream).sum())

    step = _FINITE_DIFFERENCE_STEP
    input_grad = torch.zeros_like(inputs)
    for index in range(inputs.numel()):
        shift = torch.zeros_like(inputs).flatten()
        shift[index] = step
        shift = shift.view_as(inputs)
        input_grad.view(-1)[index] = (
            scalar(inputs + shift) - scalar(inputs - shift)
        ) / (2 * step)

    original = layer.weight.detach().clone()
    weight_grad = torch.zeros_like(original)
    for index in range(original.numel()):
        layer.weight.data = original.clone()
        layer.weight.data[index] += step
        plus = scalar(inputs)
        layer.weight.data = original.clone()
        layer.weight.data[index] -= step
        minus = scalar(inputs)
        weight_grad[index] = (plus - minus) / (2 * step)
    layer.weight.data = original

    return input_grad, weight_grad


class TestQiskitAnsatzConstruction:
    """Validates the static shape and parameter contract of the ansatz."""

    def test_wire_count_gate_count_and_parameter_counts(self) -> None:
        """Asserts qubit count, per-layer gate counts, theta/gamma/beta
        parameter vector sizes, and observable count match the Bhowmik et
        al. (2025) formulation for a fixed configuration."""
        n_qubits, n_layers = 6, 4
        circuit, input_params, weight_params, observables = build_qiskit_ansatz(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        assert circuit.num_qubits == n_qubits
        assert len(input_params) == n_qubits

        theta, gamma, beta = weight_params
        assert len(theta) == n_layers * n_qubits
        assert len(gamma) == n_layers * n_qubits
        assert len(beta) == n_layers * n_qubits

        assert len(observables) == n_qubits
        for observable in observables:
            assert observable.num_qubits == n_qubits

    def test_observables_are_single_qubit_pauli_z(self) -> None:
        """Asserts each observable is a weight-one Pauli-Z string, isolating
        exactly one qubit's Z expectation per observable."""
        n_qubits = 4
        _, _, _, observables = build_qiskit_ansatz(n_qubits=n_qubits, n_layers=1)

        for observable in observables:
            label = observable.paulis.to_labels()[0]
            assert label.count("Z") == 1
            assert label.count("I") == n_qubits - 1


class TestQiskitAnsatzDynamicConfiguration:
    """Validates the ansatz adapts to varying qubit and layer counts."""

    def test_parameter_counts_scale_with_qubits_and_layers(self) -> None:
        """Asserts theta/gamma/beta vector lengths and observable count
        scale linearly with `n_qubits` and `n_layers` across a grid of
        representative configurations."""
        for n_qubits in (3, 6, 10):
            for n_layers in (2, 3, 4):
                circuit, input_params, weight_params, observables = build_qiskit_ansatz(
                    n_qubits=n_qubits, n_layers=n_layers
                )

                assert circuit.num_qubits == n_qubits
                assert len(input_params) == n_qubits
                assert len(observables) == n_qubits

                for vector in weight_params:
                    assert len(vector) == n_layers * n_qubits


class TestQiskitQuantumLayerForward:
    """Validates the broadcast forward pass against an exact reference."""

    def test_forward_matches_exact_statevector_expectations(self) -> None:
        """Asserts the single-broadcast-PUB forward pass equals a per-sample
        exact `Statevector` evaluation, which pins the weight layout, the
        qubit ordering of the observables, and the `[-1, 1]` value range."""
        n_qubits, n_layers = 3, 2
        layer = _hadamard_layer(n_qubits, n_layers, seed=0)

        inputs = torch.randn(4, n_qubits)
        outputs = layer(inputs)

        assert outputs.shape == (4, n_qubits)
        assert torch.allclose(
            outputs, _exact_expectations(layer, inputs), atol=_EXPECTATION_TOLERANCE
        )


class TestQiskitQuantumLayerSpsaBackward:
    """Validates the loss-level SPSA backward pass of the autograd function."""

    def test_averaged_spsa_gradient_points_along_exact_gradient(self) -> None:
        """Averages many SPSA backward passes (unbiased, so the mean converges
        to the true gradient) and asserts the cosine similarity with the exact
        finite-difference gradient of the same scalar loss, for both the
        circuit weights and the upstream inputs."""
        n_qubits, n_layers, batch = 2, 1, 8
        layer = _hadamard_layer(n_qubits, n_layers, seed=0)
        inputs = torch.randn(batch, n_qubits, requires_grad=True)
        upstream = torch.randn(batch, n_qubits)

        exact_input_grad, exact_weight_grad = _finite_difference_gradients(
            layer, inputs.detach(), upstream
        )

        loss = (layer(inputs) * upstream).sum()
        n_draws = 60
        for _ in range(n_draws):
            loss.backward(retain_graph=True)

        assert inputs.grad is not None
        assert layer.weight.grad is not None
        spsa = torch.cat([inputs.grad.flatten(), layer.weight.grad.flatten()])
        exact = torch.cat([exact_input_grad.flatten(), exact_weight_grad.flatten()])
        cosine = torch.nn.functional.cosine_similarity(spsa, exact, dim=0)

        assert cosine > _MIN_SPSA_COSINE_SIMILARITY

    def test_perturbations_are_reproducible_from_the_seed(self) -> None:
        """Asserts equal seeds give identical gradients and different seeds
        give different ones, so SPSA noise follows the run seed and not the
        global RNG state."""
        n_qubits, n_layers = 3, 1
        inputs = torch.randn(4, n_qubits)

        def weight_gradient(seed: int) -> torch.Tensor:
            layer = _hadamard_layer(n_qubits, n_layers, seed=seed)
            layer.weight.data = torch.linspace(-1.0, 1.0, layer.weight.numel())
            layer(inputs).sum().backward()
            assert layer.weight.grad is not None
            return layer.weight.grad

        assert torch.equal(weight_gradient(seed=7), weight_gradient(seed=7))
        assert not torch.equal(weight_gradient(seed=7), weight_gradient(seed=8))


class TestQiskitQuantumClassifierHeadEndToEnd:
    """Validates the full DQN head's forward/backward pass on cached features."""

    def test_forward_backward_pass_on_cached_embeddings(self) -> None:
        """Runs a forward pass and BCE-with-logits backward pass over a
        batch of cached feature embeddings, asserting gradients populate the
        classical pre-net, post-net, and quantum circuit weights."""
        n_qubits, n_layers = 3, 1
        head = QiskitQuantumClassifierHead(
            in_features=16,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        x = torch.randn(4, 16)
        targets = torch.randint(0, 2, (4, 1)).float()

        logits = head(x)
        assert logits.shape == (4, 1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        assert head.pre_net.weight.grad is not None
        assert head.post_net.weight.grad is not None
        assert head.qnn.weight.grad is not None


class TestDefaultEstimatorParity:
    """Validates the default Aer estimator against the reference estimator."""

    def test_default_estimator_is_aer_and_matches_statevector_estimator(self) -> None:
        """Asserts the default estimator is Aer's and that it returns the same
        expectation values as `StatevectorEstimator` on identical parameters,
        so a qiskit-aer upgrade cannot silently change the forward pass."""
        n_qubits, n_layers, batch = 4, 2, 8
        circuit, input_params, weight_vectors, observables = _hadamard_ansatz(
            n_qubits=n_qubits, n_layers=n_layers
        )
        parameters = (*input_params, *(p for vec in weight_vectors for p in vec))

        default_runner = QiskitExpectationRunner(circuit, observables, parameters)
        reference_runner = QiskitExpectationRunner(
            circuit, observables, parameters, estimator=StatevectorEstimator()
        )

        values = torch.rand(batch, len(parameters), dtype=torch.float64) * 2 * math.pi
        assert isinstance(default_runner.estimator, AerEstimator)
        assert torch.allclose(
            default_runner.expectation_values(values),
            reference_runner.expectation_values(values),
            atol=_PARITY_TOLERANCE,
        )


class _CountingEstimator(StatevectorEstimator):
    """Reference estimator that counts how many times it is run."""

    def __init__(self) -> None:
        super().__init__()
        self.run_calls = 0

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        self.run_calls += 1
        return super().run(pubs, precision=precision)


class TestQiskitQuantumClassifierHeadEstimatorInjection:
    """Validates the head evaluates its circuit on the injected estimator."""

    def test_injected_estimator_runs_one_pub_per_pass(self) -> None:
        """Asserts a forward and backward pass use the injected estimator
        exactly twice: one broadcast PUB for the forward pass and one for the
        SPSA backward pass, regardless of batch size and qubit count."""
        estimator = _CountingEstimator()
        head = QiskitQuantumClassifierHead(
            in_features=16, n_qubits=3, n_layers=1, estimator=estimator
        )

        head(torch.randn(4, 16)).sum().backward()

        assert estimator.run_calls == 2


class TestQiskitQuantumClassifierHeadGlorotInit:
    """Validates that Glorot init isolates classical layers from the circuit."""

    def test_classical_layers_reinit_while_quantum_weights_preserved(self) -> None:
        """Asserts that pre-net and post-net weights change under Glorot
        init while the quantum circuit weights remain untouched and bounded
        in [-pi, pi]."""
        head = QiskitQuantumClassifierHead(in_features=16, n_qubits=3, n_layers=1)

        pre_net_weight_before = head.pre_net.weight.clone()
        post_net_weight_before = head.post_net.weight.clone()
        qnn_weight_before = head.qnn.weight.clone()

        head.apply(QiskitQuantumClassifierHead.apply_glorot_init)

        assert not torch.equal(head.pre_net.weight, pre_net_weight_before)
        assert not torch.equal(head.post_net.weight, post_net_weight_before)
        assert torch.all(head.pre_net.bias == 0.0)
        assert torch.all(head.post_net.bias == 0.0)

        assert torch.equal(head.qnn.weight, qnn_weight_before)
        assert torch.all(head.qnn.weight >= -math.pi)
        assert torch.all(head.qnn.weight <= math.pi)
