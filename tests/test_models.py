import pytest
import torch

from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)


@pytest.fixture
def batch_size() -> int:
    """Fixture to provide a consistent batch size across tests."""
    return 4


def test_feature_extractor_output_shape(batch_size: int) -> None:
    """
    Ensures the LeNet backbone correctly processes a 128x128 image
    and outputs the expected [Batch, Channels, Height, Width] feature map.
    """
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    extractor = LeNetFeatureExtractor()

    output = extractor(dummy_input)

    # Expected shape based on 4 Convs and 3 MaxPools is [Batch, 64, 7, 7]
    expected_shape = (batch_size, 64, 7, 7)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}",
    )


def test_classifier_head_output_shape_and_bounds(batch_size: int) -> None:
    """
    Ensures the Dense head accepts the flattened features and outputs
    a valid probability bounded between 0.0 and 1.0.
    """
    # 64 channels * 7 height * 7 width = 3164 features
    in_features = 3136
    dummy_flattened_features = torch.randn(batch_size, in_features)
    head = ClassicalClassifierHead(in_features=in_features)

    output = head(dummy_flattened_features)

    # Shape must be [Batch, 1]
    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    # Sigmoid activation must bound outputs between 0 and 1
    assert torch.all(output >= 0.0), "Output contains values less than 0.0"
    assert torch.all(output <= 1.0), "Output contains values greater than 1.0"


def test_dementia_classifier_integration(batch_size: int) -> None:
    """
    Ensures the orchestrator correctly glues the extractor and head together,
    handling the implicit flattening of the 2D feature map.
    """
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    extractor = LeNetFeatureExtractor()
    head = ClassicalClassifierHead(in_features=3136)
    model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)

    output = model(dummy_input)

    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, (
        f"Integration failed. Expected {expected_shape}, got {output.shape}"
    )
