"""Unit tests for feature embedding caching and in-memory datasets.

This module validates the functionality of `CachedEmbeddingDataset`,
`FeatureCacheManager`, and model assembly utilities for transfer learning.
"""

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.data.embedding_cache import (
    CachedEmbeddingDataset,
    FeatureCacheManager,
)
from dementia_boost.models.builder import assemble_dementia_classifier
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import QuantumClassifierHead

TEST_NUM_SAMPLES: int = 16
TEST_FEATURE_DIM: int = 2304
TEST_BATCH_SIZE: int = 4
TEST_IMAGE_SIZE: int = 128
TEST_IMAGE_CHANNELS: int = 1
TEST_NUM_IMAGES: int = 12
TEST_N_QUBITS: int = 2
TEST_N_LAYERS: int = 1


@pytest.fixture
def sample_feature_data() -> tuple[Tensor, Tensor]:
    """Fixture providing dummy feature vectors and binary target labels.

    Returns:
        A tuple of feature tensor (16, 2304) and label tensor (16, 1).
    """
    features = torch.randn(TEST_NUM_SAMPLES, TEST_FEATURE_DIM)
    labels = torch.randint(0, 2, (TEST_NUM_SAMPLES, 1)).float()
    return features, labels


@pytest.fixture
def mock_image_dataloader() -> DataLoader:
    """Fixture providing a mock DataLoader with raw image tensors and labels.

    Returns:
        A DataLoader yielding batches of shape (4, 1, 128, 128) and (4, 1).
    """
    images = torch.randn(
        TEST_NUM_IMAGES,
        TEST_IMAGE_CHANNELS,
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
    )
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]] * 3)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


def test_cached_embedding_dataset_len_and_getitem(
    sample_feature_data: tuple[Tensor, Tensor],
) -> None:
    """Validates length and indexing of CachedEmbeddingDataset."""
    features, labels = sample_feature_data
    dataset = CachedEmbeddingDataset(features=features, labels=labels)

    assert len(dataset) == TEST_NUM_SAMPLES

    feat_item, label_item = dataset[0]
    assert feat_item.shape == (TEST_FEATURE_DIM,)
    assert label_item.shape == (1,)
    assert torch.equal(feat_item, features[0])
    assert torch.equal(label_item, labels[0])


def test_extract_features_shape_and_values(
    mock_image_dataloader: DataLoader,
) -> None:
    """Validates feature extraction from LeNetFeatureExtractor on mock images."""
    device = torch.device("cpu")
    extractor = LeNetFeatureExtractor()

    features, labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor,
        data_loader=mock_image_dataloader,
        device=device,
    )

    assert features.shape == (TEST_NUM_IMAGES, TEST_FEATURE_DIM)
    assert labels.shape == (TEST_NUM_IMAGES, 1)

    first_batch_images, _ = next(iter(mock_image_dataloader))
    with torch.no_grad():
        expected_first_batch_features = extractor(first_batch_images).flatten(
            start_dim=1
        )

    assert torch.allclose(
        features[:TEST_BATCH_SIZE],
        expected_first_batch_features,
        atol=1e-5,
    )


def test_create_cached_loader(sample_feature_data: tuple[Tensor, Tensor]) -> None:
    """Validates DataLoader generation from cached tensors."""
    features, labels = sample_feature_data
    loader = FeatureCacheManager.create_cached_loader(
        features=features,
        labels=labels,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    )

    assert isinstance(loader, DataLoader)
    batch_features, batch_labels = next(iter(loader))
    assert batch_features.shape == (TEST_BATCH_SIZE, TEST_FEATURE_DIM)
    assert batch_labels.shape == (TEST_BATCH_SIZE, 1)


def test_cache_disk_serialization(
    tmp_path: Path,
    sample_feature_data: tuple[Tensor, Tensor],
) -> None:
    """Validates saving and loading feature tensors to and from disk."""
    features, labels = sample_feature_data
    save_file = str(tmp_path / "cached_features.pt")

    FeatureCacheManager.save_cache(
        features=features,
        labels=labels,
        save_path=save_file,
    )

    assert Path(save_file).exists()

    loaded_features, loaded_labels = FeatureCacheManager.load_cache(
        cache_path=save_file
    )

    assert torch.equal(features, loaded_features)
    assert torch.equal(labels, loaded_labels)


def test_heads_forward_with_cached_features(
    sample_feature_data: tuple[Tensor, Tensor],
) -> None:
    """Validates direct forward pass through heads using cached feature vectors."""
    features, _ = sample_feature_data
    batch_features = features[:TEST_BATCH_SIZE]

    classical_head = ClassicalClassifierHead(
        in_features=TEST_FEATURE_DIM,
        use_sigmoid=False,
    )
    classical_output = classical_head(batch_features)
    assert classical_output.shape == (TEST_BATCH_SIZE, 1)

    quantum_head = QuantumClassifierHead(
        in_features=TEST_FEATURE_DIM,
        n_qubits=TEST_N_QUBITS,
        n_layers=TEST_N_LAYERS,
    )
    quantum_output = quantum_head(batch_features)
    assert quantum_output.shape == (TEST_BATCH_SIZE, 1)


def test_assemble_dementia_classifier() -> None:
    """Validates assembling a DementiaClassifier from separate extractor and head."""
    extractor = LeNetFeatureExtractor()
    head = ClassicalClassifierHead(
        in_features=TEST_FEATURE_DIM,
        use_sigmoid=False,
    )

    model = assemble_dementia_classifier(
        feature_extractor=extractor,
        classifier_head=head,
    )

    assert isinstance(model, DementiaClassifier)
    model.eval()
    extractor.eval()
    head.eval()

    dummy_input = torch.randn(
        2,
        TEST_IMAGE_CHANNELS,
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
    )

    with torch.no_grad():
        combined_output = model(dummy_input)
        manual_output = head(extractor(dummy_input))

    assert torch.allclose(combined_output, manual_output, atol=1e-5)
