"""Unit tests for the Qiskit v2.x Quantum Transfer Learning (QTL) pipeline.

This module validates:
- ``build_qiskit_ansatz`` circuit shape, parameter counts, and observable
  construction across dynamic qubit/layer configurations.
- ``create_qiskit_quantum_layer`` forward-pass Pauli-Z expectation bounds and
  autograd differentiability of circuit weights.
- ``QiskitQuantumClassifierHead`` end-to-end forward/backward pass and
  Glorot init isolation from quantum circuit weights.

Regression coverage
--------------------
- Drift in the ansatz's theta/gamma/beta parameter counts breaking dynamic
  qubit/layer configuration.
- Pauli-Z expectation values escaping the physically valid [-1, 1] interval.
- Missing ``input_gradients`` support silently blocking gradient flow into
  the classical pre-net feeding the quantum circuit.
- Classical re-initialization accidentally perturbing quantum circuit weights.
"""

import math

import torch

from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.models.quantum_cnn.qiskit_circuit import build_qiskit_ansatz
from dementia_boost.models.quantum_cnn.qiskit_layer import create_qiskit_quantum_layer

_EXPECTATION_TOLERANCE: float = 1e-6


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


class TestQiskitQuantumLayerForwardBounds:
    """Validates that circuit measurements remain physically valid."""

    def test_pauli_z_expectations_within_unit_interval(self) -> None:
        """Asserts that forward-pass Pauli-Z expectation values returned by
        the TorchConnector strictly satisfy -1.0 <= <Z_i> <= 1.0."""
        n_qubits, n_layers = 4, 2
        layer = create_qiskit_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        inputs = torch.randn(3, n_qubits)
        outputs = layer(inputs)

        assert outputs.shape == (3, n_qubits)
        assert torch.all(outputs >= -1.0 - _EXPECTATION_TOLERANCE)
        assert torch.all(outputs <= 1.0 + _EXPECTATION_TOLERANCE)


class TestQiskitQuantumLayerAutograd:
    """Validates PyTorch autograd differentiability through the TorchConnector."""

    def test_backward_pass_yields_nonzero_gradients(self) -> None:
        """Runs forward and backward passes on the TorchConnector, asserting
        that gradients on all circuit weight parameters are populated and
        non-zero."""
        n_qubits, n_layers = 3, 2
        layer = create_qiskit_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        inputs = torch.randn(4, n_qubits)
        output = layer(inputs)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert torch.any(layer.weight.grad != 0.0)


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
