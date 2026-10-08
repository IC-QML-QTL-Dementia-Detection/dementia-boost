"""Unit tests for model backbones, classification heads, and autograd flow.

This module validates:
- ``LeNetFeatureExtractor`` intermediate layer shapes and gradient propagation.
- ``ClassicalClassifierHead`` logit/sigmoid output contracts and Glorot init.
- ``DementiaClassifier`` backbone freezing contracts under transfer learning.
- The PennyLane variational ansatz's angle scaling bounds, Pauli-Z expectation
  bounds, and autograd differentiability.
- ``PennylaneQuantumClassifierHead`` Glorot init isolation from quantum circuit weights.

Regression coverage
-------------------
- Silent shape drift in the convolutional backbone breaking downstream heads.
- Frozen backbones accidentally receiving gradient updates during transfer
  learning, corrupting pre-trained spatial representations.
- Angle embedding inputs escaping the [-pi/2, pi/2] domain under extreme
  activations, invalidating the physical rotation semantics.
- Vanishing gradients through the quantum circuit blocking training.
- Classical re-initialization accidentally perturbing quantum circuit weights.
"""

import math

import pennylane as qml
import torch
import torch.nn as nn

from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import PennylaneQuantumClassifierHead
from dementia_boost.models.quantum_cnn.pennylane_circuit import (
    _build_custom_ansatz,
    create_quantum_layer,
    resolve_quantum_device,
)

_EXPECTATION_TOLERANCE: float = 1e-4
_ANGLE_TOLERANCE: float = 1e-6
_MIN_GRADIENT: float = 1e-6
_MIN_SPREAD: float = 1e-3


def _random_angles(batch: int, n_qubits: int) -> torch.Tensor:
    """Draws embedding angles over the range the pre-net can produce.

    Args:
        batch: Number of samples.
        n_qubits: Number of qubits (columns).

    Returns:
        A `(batch, n_qubits)` tensor uniform in `[-pi/2, pi/2]`.
    """
    return (torch.rand(batch, n_qubits) - 0.5) * math.pi


class TestLeNetFeatureExtractor:
    """Validates the convolutional backbone's shape contract and trainability."""

    _BLOCK_END_INDICES: tuple[int, ...] = (2, 5, 8, 10)
    _BATCH_SIZE: int = 2

    def _expected_block_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Builds the expected output shape at the end of each conv block.

        Returns:
            A tuple of four `(Batch, Channels, Height, Width)` shape tuples,
            one per convolutional block boundary.
        """
        return (
            (self._BATCH_SIZE, 8, 62, 62),
            (self._BATCH_SIZE, 16, 27, 27),
            (self._BATCH_SIZE, 32, 9, 9),
            (self._BATCH_SIZE, 64, 6, 6),
        )

    def test_layer_shapes_traced_through_each_block(self) -> None:
        """Asserts that each convolutional block emits its documented shape."""
        extractor = LeNetFeatureExtractor()
        x = torch.randn(self._BATCH_SIZE, 1, 128, 128)
        expected_shapes = self._expected_block_shapes()

        block_idx = 0
        out = x
        for i, layer in enumerate(extractor.features):
            out = layer(out)
            if i == self._BLOCK_END_INDICES[block_idx]:
                assert out.shape == expected_shapes[block_idx], (
                    f"Block {block_idx} expected {expected_shapes[block_idx]}, "
                    f"got {tuple(out.shape)}"
                )
                block_idx += 1

        assert block_idx == len(expected_shapes)

    def test_gradient_propagation_to_all_conv_weights(self) -> None:
        """Asserts that backward pass populates non-zero gradients on all
        four convolutional weight tensors."""
        extractor = LeNetFeatureExtractor()
        x = torch.randn(self._BATCH_SIZE, 1, 128, 128)

        output = extractor(x)
        loss = output.sum()
        loss.backward()

        conv_layers = [
            layer for layer in extractor.features if isinstance(layer, nn.Conv2d)
        ]
        assert len(conv_layers) == 4

        for conv in conv_layers:
            assert conv.weight.grad is not None
            assert torch.any(conv.weight.grad != 0.0)


class TestClassicalClassifierHead:
    """Validates output contracts and initialization of the dense head."""

    _IN_FEATURES: int = 2304
    _BATCH_SIZE: int = 4

    def test_logits_unconstrained_and_sigmoid_bounded(self) -> None:
        """Asserts that `use_sigmoid=False` yields unconstrained logits while
        `use_sigmoid=True` strictly bounds outputs to (0.0, 1.0)."""
        torch.manual_seed(42)
        x = torch.randn(self._BATCH_SIZE, self._IN_FEATURES) * 10.0

        logits_head = ClassicalClassifierHead(
            in_features=self._IN_FEATURES,
            use_sigmoid=False,
        )
        logits_head.eval()
        sigmoid_head = ClassicalClassifierHead(
            in_features=self._IN_FEATURES,
            use_sigmoid=True,
        )
        sigmoid_head.eval()

        logits = logits_head(x)
        probabilities = sigmoid_head(x)

        assert torch.any(torch.abs(logits) > 1.0)
        assert torch.all(probabilities > 0.0)
        assert torch.all(probabilities < 1.0)

    def test_glorot_init_reinitializes_weights_and_zeros_biases(self) -> None:
        """Asserts that Glorot Uniform init changes linear weights from their
        default PyTorch initialization and resets biases to zero."""
        head = ClassicalClassifierHead(in_features=self._IN_FEATURES)
        linear_layers = [m for m in head.classifier if isinstance(m, nn.Linear)]
        weights_before = [layer.weight.clone() for layer in linear_layers]

        head.apply(ClassicalClassifierHead.apply_glorot_init)

        for layer, weight_before in zip(linear_layers, weights_before, strict=True):
            assert not torch.equal(layer.weight, weight_before)
            assert torch.all(layer.bias == 0.0)


class TestDementiaClassifierFreezingContract:
    """Validates that frozen backbones receive zero gradient updates."""

    def test_frozen_backbone_receives_no_gradients(self) -> None:
        """Asserts that a frozen feature extractor's parameters keep `.grad`
        as `None` after backward, while the classifier head is updated."""
        extractor = LeNetFeatureExtractor()
        for param in extractor.parameters():
            param.requires_grad = False

        head = ClassicalClassifierHead(use_sigmoid=False)
        model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)

        x = torch.randn(2, 1, 128, 128)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for param in model.feature_extractor.parameters():
            assert param.grad is None

        for param in model.classifier_head.parameters():
            assert param.grad is not None


class TestPennylaneQuantumClassifierHeadAngleScaling:
    """Validates the pre-net and angle scaling bound the quantum inputs."""

    def test_extreme_inputs_stay_within_rotation_bounds(self) -> None:
        """Passes extreme-magnitude inputs through the pre-net and angle
        scaling, asserting outputs remain strictly in [-pi/2, pi/2]."""
        head = PennylaneQuantumClassifierHead(
            in_features=16,
            n_qubits=3,
            n_layers=1,
            quantum_device="default.qubit",
        )

        for magnitude in (1000.0, -1000.0):
            x = torch.full((2, 16), magnitude)
            flattened = head.flatten(x)
            projected = head.pre_net(flattened)
            scaled = torch.tanh(projected) * head.ANGLE_SCALING_FACTOR

            assert torch.all(scaled <= math.pi / 2.0 + _ANGLE_TOLERANCE)
            assert torch.all(scaled >= -math.pi / 2.0 - _ANGLE_TOLERANCE)


class TestQuantumAnsatzPauliZBounds:
    """Validates that circuit measurements remain physically valid."""

    def test_pauli_z_expectations_within_unit_interval(self) -> None:
        """Asserts that Pauli-Z expectation values returned by the custom
        ansatz strictly satisfy -1.0 <= <Z_i> <= 1.0 under extreme
        parameter magnitudes."""
        n_qubits = 4
        n_layers = 2
        device = resolve_quantum_device(
            n_qubits=n_qubits,
            quantum_device="default.qubit",
        )

        @qml.qnode(device, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            return _build_custom_ansatz(inputs, weights, n_qubits, n_layers)

        inputs = torch.randn(3, n_qubits) * 100.0
        weights = torch.randn(n_layers, 3, n_qubits) * 100.0

        expectations = torch.stack(circuit(inputs, weights))

        assert torch.all(expectations >= -1.0 - _EXPECTATION_TOLERANCE)
        assert torch.all(expectations <= 1.0 + _EXPECTATION_TOLERANCE)


class TestQuantumLayerAutograd:
    """Validates PyTorch autograd differentiability through the QNode."""

    def test_backward_pass_yields_nonzero_gradients(self) -> None:
        """Runs forward and backward passes on the `TorchLayer`, asserting
        that adjoint differentiation gives gradients with respect to the
        circuit weights and the inputs that are clearly above float noise.

        A constant circuit can still return gradients around 1e-17, so the
        check compares against a threshold instead of against exact zero."""
        n_qubits = 3
        n_layers = 2
        layer = create_quantum_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            quantum_device="default.qubit",
        )

        inputs = _random_angles(8, n_qubits).requires_grad_()
        output = layer(inputs)
        loss = (output * torch.randn_like(output)).sum()
        loss.backward()

        assert layer.weights.grad is not None
        assert inputs.grad is not None
        assert layer.weights.grad.abs().max() > _MIN_GRADIENT
        assert inputs.grad.abs().max() > _MIN_GRADIENT


class TestQuantumLayerInputDependence:
    """Guards against the circuit degenerating into a constant function.

    The paper's gate sequence applied to `|0...0>` keeps the state in the
    `|0...0>` subspace, so every `<Z_i>` equals 1 for all inputs and weights.
    """

    def test_every_expectation_value_varies_with_the_inputs(self) -> None:
        """Asserts that each qubit's `<Z_i>` takes clearly different values
        across a batch of random inputs at fixed weights."""
        n_qubits = 4
        layer = create_quantum_layer(
            n_qubits=n_qubits, n_layers=2, quantum_device="default.qubit"
        )

        with torch.no_grad():
            outputs = layer(_random_angles(32, n_qubits))

        assert torch.all(outputs.std(dim=0) > _MIN_SPREAD)

    def test_expectation_values_vary_with_the_weights(self) -> None:
        """Asserts that two different weight draws give clearly different
        outputs on the same inputs, so the trainable weights matter."""
        n_qubits = 4
        inputs = _random_angles(8, n_qubits)

        def evaluate(seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            layer = create_quantum_layer(
                n_qubits=n_qubits, n_layers=2, quantum_device="default.qubit"
            )
            with torch.no_grad():
                return layer(inputs)

        assert (evaluate(0) - evaluate(1)).abs().max() > _MIN_SPREAD


class TestPennylaneQuantumClassifierHeadGlorotInit:
    """Validates that Glorot init isolates classical layers from the QNode."""

    def test_classical_layers_reinit_while_quantum_weights_preserved(self) -> None:
        """Asserts that pre-net and post-net weights change under Glorot
        init while the quantum circuit weights remain untouched and bounded
        in [-pi, pi]."""
        head = PennylaneQuantumClassifierHead(
            in_features=16,
            n_qubits=3,
            n_layers=1,
            quantum_device="default.qubit",
        )

        pre_net_weight_before = head.pre_net.weight.clone()
        post_net_weight_before = head.post_net.weight.clone()
        qnn_weights_before = head.qnn.weights.clone()

        head.apply(PennylaneQuantumClassifierHead.apply_glorot_init)

        assert not torch.equal(head.pre_net.weight, pre_net_weight_before)
        assert not torch.equal(head.post_net.weight, post_net_weight_before)
        assert torch.all(head.pre_net.bias == 0.0)
        assert torch.all(head.post_net.bias == 0.0)

        assert torch.equal(head.qnn.weights, qnn_weights_before)
        assert torch.all(head.qnn.weights >= -math.pi)
        assert torch.all(head.qnn.weights <= math.pi)
