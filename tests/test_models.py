import pennylane as qml
import pytest
import torch

from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import (
    DEFAULT_QUANTUM_DEVICE,
    FALLBACK_QUANTUM_DEVICE,
    QuantumClassifierHead,
    resolve_quantum_device,
)


@pytest.fixture
def batch_size() -> int:
    """Fixture to provide a consistent batch size across tests."""
    return 4


def test_feature_extractor_output_shape(batch_size: int) -> None:
    """Ensures the LeNet backbone correctly processes a 128x128 image
    and outputs the expected [Batch, Channels, Height, Width] feature map.
    """
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    extractor = LeNetFeatureExtractor()

    output = extractor(dummy_input)

    expected_shape = (batch_size, 64, 6, 6)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )


def test_classifier_head_output_shape_and_bounds(batch_size: int) -> None:
    """Ensures the Dense head accepts the flattened features and outputs
    a valid probability bounded between 0.0 and 1.0.
    """
    in_features = 2304
    dummy_flattened_features = torch.randn(batch_size, in_features)
    head = ClassicalClassifierHead(in_features=in_features)

    output = head(dummy_flattened_features)

    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    assert torch.all(output >= 0.0), "Output contains values less than 0.0"
    assert torch.all(output <= 1.0), "Output contains values greater than 1.0"


def test_dementia_classifier_integration(batch_size: int) -> None:
    """Ensures the orchestrator correctly glues the extractor and head together,
    handling the implicit flattening of the 2D feature map.
    """
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    extractor = LeNetFeatureExtractor()
    head = ClassicalClassifierHead(in_features=2304)
    model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)

    output = model(dummy_input)

    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Integration failed. Expected {expected_shape}, got {output.shape}"
    )


def test_resolve_quantum_device() -> None:
    """Verifies that quantum device resolution supports defaults, explicit names,
    fallback for non-existent backends, and pre-instantiated device objects.
    """
    dev_default = resolve_quantum_device(n_qubits=4)
    assert dev_default.name in (DEFAULT_QUANTUM_DEVICE, FALLBACK_QUANTUM_DEVICE)

    dev_explicit = resolve_quantum_device(n_qubits=4, quantum_device="default.qubit")
    assert dev_explicit.name == "default.qubit"

    dev_fallback = resolve_quantum_device(
        n_qubits=4,
        quantum_device="invalid_non_existent_device_backend",
    )
    assert dev_fallback.name == FALLBACK_QUANTUM_DEVICE

    custom_dev = qml.device("default.qubit", wires=4)
    dev_passed = resolve_quantum_device(n_qubits=4, quantum_device=custom_dev)
    assert dev_passed is custom_dev


def test_quantum_classifier_head_output_shape(batch_size: int) -> None:
    """Ensures the QuantumClassifierHead processes flattened features and outputs
    raw logits with the expected shape [Batch, 1].
    """
    in_features = 2304
    n_qubits = 4
    n_layers = 2
    dummy_features = torch.randn(batch_size, in_features)

    head = QuantumClassifierHead(
        in_features=in_features,
        n_qubits=n_qubits,
        n_layers=n_layers,
        quantum_device="default.qubit",
    )
    output = head(dummy_features)

    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )


def test_dementia_classifier_quantum_integration(batch_size: int) -> None:
    """Ensures the DementiaClassifier orchestrator operates seamlessly with
    a QuantumClassifierHead.
    """
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    extractor = LeNetFeatureExtractor()
    head = QuantumClassifierHead(
        in_features=2304,
        n_qubits=4,
        n_layers=2,
        quantum_device="default.qubit",
    )
    model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)

    output = model(dummy_input)

    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Integration failed. Expected {expected_shape}, got {output.shape}"
    )
