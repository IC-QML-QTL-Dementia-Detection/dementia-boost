"""Unit tests for model backbones, classification heads, and autograd flow.

This module validates:
- ``LeNetFeatureExtractor`` intermediate layer shapes and gradient propagation.
- ``ClassicalClassifierHead`` logit/sigmoid output contracts and Glorot init.
- The registry of classical heads, the circuit-free control head, and the head
  rebuilt from a run's spec.
- ``DementiaClassifier`` backbone freezing contracts under transfer learning.
- The quantum head's angle scaling bounds.

The quantum circuit itself (input and weight dependence, gradients, Glorot init
isolation, agreement with an exact simulation) is tested in
``test_qiskit_qtl.py`` and ``test_circuit_acceptance.py``.

Regression coverage
-------------------
- Silent shape drift in the convolutional backbone breaking downstream heads.
- Frozen backbones accidentally receiving gradient updates during transfer
  learning, corrupting pre-trained spatial representations.
- Angle embedding inputs escaping the [-pi/2, pi/2] domain under extreme
  activations, invalidating the physical rotation semantics.
- The CTL evaluation built the paper's head whatever the spec said, so a
  checkpoint of another head would not load.
"""

import math

import pytest
import torch
import torch.nn as nn

from dementia_boost.models.classical_cnn import (
    CLASSICAL_HEADS,
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
    TanhBottleneckHead,
    build_classical_head,
    classical_head_for,
    stored_hidden_features,
)
from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead

_ANGLE_TOLERANCE: float = 1e-6


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


class TestClassicalHeadRegistry:
    """Validates the registered classical heads and their link to the spec."""

    _IN_FEATURES: int = 2304
    _BATCH_SIZE: int = 4

    def test_paper_head_is_the_ctl_head(self) -> None:
        """`paper` builds the base paper's head: 2304 -> 5, dropout, ReLU, 5 -> 1."""
        head = build_classical_head("paper")
        assert isinstance(head, ClassicalClassifierHead)
        layers = [type(layer) for layer in head.classifier]
        assert layers == [nn.Flatten, nn.Linear, nn.Dropout, nn.ReLU, nn.Linear]
        assert head.classifier[1].out_features == 5

    def test_tanh_bottleneck_is_the_quantum_head_without_the_circuit(self) -> None:
        """`tanh_bottleneck` has the quantum head's classical layers (2304 -> 6,
        tanh times pi/2, 6 -> 1), no dropout, and one logit per sample."""
        head = build_classical_head("tanh_bottleneck")
        assert isinstance(head, TanhBottleneckHead)
        assert head.pre_net.in_features == self._IN_FEATURES
        assert head.pre_net.out_features == QiskitQuantumClassifierHead.DEFAULT_N_QUBITS
        assert head.post_net.in_features == head.pre_net.out_features
        assert head.ANGLE_SCALING_FACTOR == (
            QiskitQuantumClassifierHead.ANGLE_SCALING_FACTOR
        )
        assert not any(isinstance(module, nn.Dropout) for module in head.modules())

        x = torch.randn(self._BATCH_SIZE, 64, 6, 6) * 1e4
        hidden = head.bottleneck(x)
        assert torch.all(torch.abs(hidden) <= math.pi / 2 + _ANGLE_TOLERANCE)
        assert head(x).shape == (self._BATCH_SIZE, 1)

    @pytest.mark.parametrize("name", sorted(CLASSICAL_HEADS))
    def test_width_is_configurable(self, name: str) -> None:
        """Every registered head takes its hidden width from `hidden_features`."""
        head = build_classical_head(name, hidden_features=8)
        widths = [m.out_features for m in head.modules() if isinstance(m, nn.Linear)]
        assert widths == [8, 1]

    def test_unknown_head_is_refused(self) -> None:
        """A mistyped name fails and lists the registered heads."""
        with pytest.raises(ValueError, match="tanh_bottleneck"):
            build_classical_head("tanh")

    @pytest.mark.parametrize(
        ("head", "hidden_features"),
        [(None, None), ("tanh_bottleneck", None), ("tanh_bottleneck", 8)],
    )
    def test_checkpoint_loads_into_the_head_rebuilt_from_its_spec(
        self, make_spec, head: str | None, hidden_features: int | None
    ) -> None:
        """The evaluation rebuilds the head from the stored spec; the weights of
        a trained head load into it with every key and shape matching. Before,
        the evaluation always built the paper's head."""
        spec = make_spec("ctl", head=head, hidden_features=hidden_features)
        trained = build_classical_head(head or "paper", hidden_features=hidden_features)

        rebuilt = classical_head_for(spec)

        assert type(rebuilt) is type(trained)
        rebuilt.load_state_dict(trained.state_dict(), strict=True)

    def test_default_width_is_stored_as_none(self) -> None:
        """Asking for a head's own width stores `None`, so one head has one ID."""
        default = TanhBottleneckHead.DEFAULT_HIDDEN_FEATURES
        assert stored_hidden_features("tanh_bottleneck", default) is None
        assert stored_hidden_features("tanh_bottleneck", None) is None
        assert stored_hidden_features("paper", default) == default

    def test_only_a_ctl_spec_has_a_classical_head(self, make_spec) -> None:
        """A quantum spec is refused."""
        with pytest.raises(ValueError, match="ctl"):
            classical_head_for(make_spec("qiskit_qtl"))


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


class TestQuantumClassifierHeadAngleScaling:
    """Validates the pre-net and angle scaling bound the quantum inputs."""

    def test_extreme_inputs_stay_within_rotation_bounds(self) -> None:
        """Passes extreme-magnitude inputs through the pre-net and angle
        scaling, asserting outputs remain strictly in [-pi/2, pi/2]."""
        head = QiskitQuantumClassifierHead(in_features=16, n_qubits=3, n_layers=1)

        for magnitude in (1000.0, -1000.0):
            x = torch.full((2, 16), magnitude)
            flattened = head.flatten(x)
            projected = head.pre_net(flattened)
            scaled = torch.tanh(projected) * head.ANGLE_SCALING_FACTOR

            assert torch.all(scaled <= math.pi / 2.0 + _ANGLE_TOLERANCE)
            assert torch.all(scaled >= -math.pi / 2.0 - _ANGLE_TOLERANCE)
