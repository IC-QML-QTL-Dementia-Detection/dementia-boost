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

Input and weight dependence of the circuit output, and the agreement of the
default Aer estimator with the reference, are checked on every registered
circuit in ``test_circuit_acceptance.py``.

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

import pytest
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import (
    PrimitiveJob,
    PrimitiveResult,
    PubResult,
    StatevectorEstimator,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPubLike
from qiskit.quantum_info import Statevector

from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.models.quantum_cnn.qiskit_circuit import build_qiskit_ansatz
from dementia_boost.models.quantum_cnn.qiskit_layer import (
    QiskitQuantumLayer,
    create_qiskit_quantum_layer,
)

_EXPECTATION_TOLERANCE: float = 1e-5
_FINITE_DIFFERENCE_STEP: float = 1e-3
_MIN_SPSA_COSINE_SIMILARITY: float = 0.9


def _layer(n_qubits: int, n_layers: int, seed: int) -> QiskitQuantumLayer:
    """Creates a seeded layer on the reference estimator, for small circuits."""
    return create_qiskit_quantum_layer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
        estimator=StatevectorEstimator(),
    )


def _exact_expectations(
    layer: QiskitQuantumLayer, inputs: torch.Tensor
) -> torch.Tensor:
    """Evaluates the layer's circuit exactly with `Statevector`, per row.

    Args:
        layer: A layer built from `build_qiskit_ansatz`, supplying the flat
            weights.
        inputs: Angle tensor of shape `(Batch, n_qubits)`.

    Returns:
        Tensor of shape `(Batch, n_qubits)` with the exact `<Z_i>` values.
    """
    circuit, input_params, weight_vectors, observables = build_qiskit_ansatz(
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


def _instructions(
    circuit: QuantumCircuit,
) -> list[tuple[str, tuple[int, ...], tuple[str, ...]]]:
    """Lists a circuit's gates as `(name, qubit indices, parameter names)`."""
    return [
        (
            instruction.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            tuple(str(param) for param in instruction.operation.params),
        )
        for instruction in circuit.data
    ]


def _angle_embedding_gates(
    n_qubits: int,
) -> list[tuple[str, tuple[int, ...], tuple[str, ...]]]:
    """Lists the gates of the `paper` embedding: Hadamards, then `RZ(x_i)`."""
    gates = [("h", (q,), ()) for q in range(n_qubits)]
    gates += [("rz", (q,), (f"x[{q}]",)) for q in range(n_qubits)]
    return gates


def _paper_layer_gates(
    n_qubits: int, layer: int
) -> list[tuple[str, tuple[int, ...], tuple[str, ...]]]:
    """Lists the gates of one layer of the base paper's block, written out by
    hand: `RZ(theta)`, the CNOT ring, `RZ(gamma)`, the CRY ring."""
    offset = layer * n_qubits
    gates = [("rz", (q,), (f"theta[{offset + q}]",)) for q in range(n_qubits)]
    gates += [("cx", (q, (q + 1) % n_qubits), ()) for q in range(n_qubits)]
    gates += [("rz", (q,), (f"gamma[{offset + q}]",)) for q in range(n_qubits)]
    gates += [
        ("cry", ((q + 1) % n_qubits, q), (f"beta[{offset + q}]",))
        for q in range(n_qubits)
    ]
    return gates


class TestQiskitAnsatzConstruction:
    """Validates the static shape and parameter contract of the ansatz."""

    def test_observables_are_single_qubit_pauli_z(self) -> None:
        """Asserts each observable is a weight-one Pauli-Z string, isolating
        exactly one qubit's Z expectation per observable."""
        n_qubits = 4
        _, _, _, observables = build_qiskit_ansatz(n_qubits=n_qubits, n_layers=1)

        for observable in observables:
            label = observable.paulis.to_labels()[0]
            assert label.count("Z") == 1
            assert label.count("I") == n_qubits - 1

    def test_paper_block_keeps_the_gate_and_weight_order(self) -> None:
        """Asserts the registered `paper` block builds exactly the base paper's
        gate sequence after the Hadamard and RZ embedding, with the weights in
        the flat `[theta, gamma, beta]` order that saved checkpoints rely on."""
        n_qubits, n_layers = 3, 2
        circuit, _, weight_params, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers, block="paper"
        )

        expected = _angle_embedding_gates(n_qubits)
        for layer in range(n_layers):
            expected += _paper_layer_gates(n_qubits, layer)

        assert _instructions(circuit) == expected
        assert [vector.name for vector in weight_params] == ["theta", "gamma", "beta"]

    def test_reupload_repeats_the_embedding_before_every_layer(self) -> None:
        """Asserts that re-uploading applies the whole embedding, Hadamard
        included, before each layer, reusing the same input parameters, so the
        input width and the weights stay those of the plain circuit."""
        n_qubits, n_layers = 3, 2
        circuit, input_params, weight_params, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers, reupload=True
        )
        _, plain_inputs, plain_weights, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers
        )

        expected = []
        for layer in range(n_layers):
            expected += _angle_embedding_gates(n_qubits)
            expected += _paper_layer_gates(n_qubits, layer)

        assert _instructions(circuit) == expected
        assert len(input_params) == len(plain_inputs)
        assert [len(v) for v in weight_params] == [len(v) for v in plain_weights]

    def test_reupload_with_one_layer_is_the_plain_circuit(self) -> None:
        """Asserts that with a single layer there is nothing to repeat."""
        reuploaded, *_ = build_qiskit_ansatz(n_qubits=3, n_layers=1, reupload=True)
        plain, *_ = build_qiskit_ansatz(n_qubits=3, n_layers=1)

        assert _instructions(reuploaded) == _instructions(plain)

    @pytest.mark.parametrize(
        ("block", "pairs"),
        [
            ("hardware_efficient_linear", [(0, 1), (1, 2)]),
            ("hardware_efficient_ring", [(0, 1), (1, 2), (2, 0)]),
            ("hardware_efficient_full", [(0, 1), (0, 2), (1, 2)]),
        ],
    )
    def test_hardware_efficient_blocks(
        self, block: str, pairs: list[tuple[int, int]]
    ) -> None:
        """Asserts `RY(theta)` and `RZ(phi)` on every qubit, then CNOTs over the
        entanglement pattern, with the weights laid out as `[theta, phi]`."""
        n_qubits, n_layers = 3, 2
        circuit, _, weight_params, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers, block=block
        )

        expected = _angle_embedding_gates(n_qubits)
        for layer in range(n_layers):
            offset = layer * n_qubits
            expected += [
                ("ry", (q,), (f"theta[{offset + q}]",)) for q in range(n_qubits)
            ]
            expected += [("rz", (q,), (f"phi[{offset + q}]",)) for q in range(n_qubits)]
            expected += [("cx", pair, ()) for pair in pairs]

        assert _instructions(circuit) == expected
        assert [vector.name for vector in weight_params] == ["theta", "phi"]

    def test_strongly_entangling_block(self) -> None:
        """Asserts `RZ(phi) RY(theta) RZ(omega)` on each qubit, then a CNOT ring
        `i -> (i + r) mod n` whose range `r` is 1 in the first layer and 2 in
        the second (PennyLane's `StronglyEntanglingLayers` rule), with the
        weights laid out as `[phi, theta, omega]`."""
        n_qubits, n_layers = 3, 2
        circuit, _, weight_params, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers, block="strongly_entangling"
        )

        expected = _angle_embedding_gates(n_qubits)
        for layer, shift in ((0, 1), (1, 2)):
            offset = layer * n_qubits
            for q in range(n_qubits):
                expected += [
                    ("rz", (q,), (f"phi[{offset + q}]",)),
                    ("ry", (q,), (f"theta[{offset + q}]",)),
                    ("rz", (q,), (f"omega[{offset + q}]",)),
                ]
            expected += [("cx", (q, (q + shift) % n_qubits), ()) for q in range(3)]

        assert _instructions(circuit) == expected
        assert [vector.name for vector in weight_params] == ["phi", "theta", "omega"]

    def test_trainable_zz_block(self) -> None:
        """Asserts trainable `RZZ(w)` on neighbouring pairs, then `RX(alpha)` on
        each qubit, with `n - 1` couplings and `n` rotations per layer, laid
        out as `[w, alpha]`."""
        n_qubits, n_layers = 3, 2
        circuit, _, weight_params, _ = build_qiskit_ansatz(
            n_qubits=n_qubits, n_layers=n_layers, block="trainable_zz"
        )

        expected = _angle_embedding_gates(n_qubits)
        for layer in range(n_layers):
            expected += [
                ("rzz", (i, i + 1), (f"w[{layer * (n_qubits - 1) + i}]",))
                for i in range(n_qubits - 1)
            ]
            expected += [
                ("rx", (q,), (f"alpha[{layer * n_qubits + q}]",))
                for q in range(n_qubits)
            ]

        assert _instructions(circuit) == expected
        assert [vector.name for vector in weight_params] == ["w", "alpha"]
        assert [len(vector) for vector in weight_params] == [4, 6]

    def test_unknown_block_is_refused(self) -> None:
        """Asserts an unregistered block name raises with the available names."""
        with pytest.raises(ValueError, match="paper"):
            build_qiskit_ansatz(n_qubits=2, n_layers=1, block="missing")


class TestQiskitAnsatzDynamicConfiguration:
    """Validates the ansatz adapts to varying qubit and layer counts."""

    def test_parameter_counts_scale_with_qubits_and_layers(self) -> None:
        """Asserts theta/gamma/beta vector lengths and observable count
        scale linearly with `n_qubits` and `n_layers` across a grid of
        representative configurations, including the paper's 6 x 4."""
        for n_qubits in (3, 6, 10):
            for n_layers in (2, 3, 4):
                circuit, input_params, weight_params, observables = build_qiskit_ansatz(
                    n_qubits=n_qubits, n_layers=n_layers
                )

                assert circuit.num_qubits == n_qubits
                assert len(input_params) == n_qubits
                assert len(observables) == n_qubits
                assert all(op.num_qubits == n_qubits for op in observables)

                assert len(weight_params) == 3
                for vector in weight_params:
                    assert len(vector) == n_layers * n_qubits


class TestQiskitQuantumLayerForward:
    """Validates the broadcast forward pass against an exact reference."""

    def test_forward_matches_exact_statevector_expectations(self) -> None:
        """Asserts the single-broadcast-PUB forward pass equals a per-sample
        exact `Statevector` evaluation, which pins the weight layout, the
        qubit ordering of the observables, and the `[-1, 1]` value range."""
        n_qubits, n_layers = 3, 2
        layer = _layer(n_qubits, n_layers, seed=0)

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
        circuit weights and the upstream inputs. Runs on the default Aer
        estimator, which the parity test pins to the reference."""
        n_qubits, n_layers, batch = 2, 1, 8
        torch.manual_seed(0)
        layer = create_qiskit_quantum_layer(
            n_qubits=n_qubits, n_layers=n_layers, seed=0
        )
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
            layer = _layer(n_qubits, n_layers, seed=seed)
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
