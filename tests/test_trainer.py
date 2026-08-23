"""Unit tests for BaselineTrainer lifecycle and checkpointing.

This module validates that BaselineTrainer supports both end-to-end full model
training on raw images and fast transfer learning on cached feature embeddings
with composite model checkpoint serialization for CTL and QTL heads.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.data.embedding_cache import FeatureCacheManager
from dementia_boost.models.builder import assemble_dementia_classifier
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import QuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.evaluator import ModelEvaluator
from dementia_boost.training.trainer import BaselineTrainer

NUM_MOCK_SAMPLES: int = 8
NUM_CHANNELS: int = 1
IMAGE_SIZE: int = 128
BATCH_SIZE: int = 4
FEATURE_DIM: int = 2304
NUM_EPOCHS: int = 2
LEARNING_RATE: float = 1e-3
STEP_SIZE: int = 5
GAMMA: float = 0.5
TEST_QTL_QUBITS: int = 2
TEST_QTL_LAYERS: int = 1


def test_baseline_trainer_with_cached_embeddings_and_save_model(
    tmp_path: Path,
) -> None:
    """Validates training a head on cached embeddings and saving assembled model."""
    device = torch.device("cpu")
    logger = setup_logger("test_trainer_ctl")
    save_dir = str(tmp_path / "checkpoints_ctl")

    extractor = LeNetFeatureExtractor().to(device)
    for param in extractor.parameters():
        param.requires_grad = False

    raw_images = torch.randn(
        NUM_MOCK_SAMPLES,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    raw_labels = torch.tensor([0.0, 1.0] * (NUM_MOCK_SAMPLES // 2))
    raw_dataset = TensorDataset(raw_images, raw_labels)
    raw_loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor,
        data_loader=raw_loader,
        device=device,
    )
    test_features, test_labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor,
        data_loader=raw_loader,
        device=device,
    )

    train_cached_loader = FeatureCacheManager.create_cached_loader(
        features=train_features,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_cached_loader = FeatureCacheManager.create_cached_loader(
        features=test_features,
        labels=test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    head = ClassicalClassifierHead(
        in_features=FEATURE_DIM,
        use_sigmoid=False,
    ).to(device)
    head.apply(ClassicalClassifierHead.apply_glorot_init)

    full_model = assemble_dementia_classifier(
        feature_extractor=extractor,
        classifier_head=head,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(head.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    trainer = BaselineTrainer(
        model=head,
        train_loader=train_cached_loader,
        test_loader=test_cached_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        save_dir=save_dir,
        save_model=full_model,
    )

    trainer.train(epochs=NUM_EPOCHS, run_id="test_run_ctl")

    saved_checkpoint_path = Path(save_dir) / "baseline_test_run_ctl.pt"
    assert saved_checkpoint_path.exists()

    state_dict = torch.load(saved_checkpoint_path, weights_only=True)
    assert any(k.startswith("feature_extractor.") for k in state_dict.keys())
    assert any(k.startswith("classifier_head.") for k in state_dict.keys())

    evaluator_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )
    evaluator = ModelEvaluator(model=evaluator_model, device=device)
    evaluator.load_weights(str(saved_checkpoint_path))

    y_true, y_prob = evaluator.predict(raw_loader)
    assert y_true.shape == (NUM_MOCK_SAMPLES,)
    assert y_prob.shape == (NUM_MOCK_SAMPLES,)
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()


def test_baseline_trainer_with_quantum_head_and_cached_embeddings(
    tmp_path: Path,
) -> None:
    """Validates training a quantum head on cached embeddings and saving model."""
    device = torch.device("cpu")
    logger = setup_logger("test_trainer_qtl")
    save_dir = str(tmp_path / "checkpoints_qtl")

    extractor = LeNetFeatureExtractor().to(device)
    for param in extractor.parameters():
        param.requires_grad = False

    raw_images = torch.randn(
        NUM_MOCK_SAMPLES,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    raw_labels = torch.tensor([0.0, 1.0] * (NUM_MOCK_SAMPLES // 2))
    raw_dataset = TensorDataset(raw_images, raw_labels)
    raw_loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor,
        data_loader=raw_loader,
        device=device,
    )
    test_features, test_labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor,
        data_loader=raw_loader,
        device=device,
    )

    train_cached_loader = FeatureCacheManager.create_cached_loader(
        features=train_features,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_cached_loader = FeatureCacheManager.create_cached_loader(
        features=test_features,
        labels=test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    qtl_head = QuantumClassifierHead(
        in_features=FEATURE_DIM,
        n_qubits=TEST_QTL_QUBITS,
        n_layers=TEST_QTL_LAYERS,
    ).to(device)
    qtl_head.apply(QuantumClassifierHead.apply_glorot_init)

    full_model = assemble_dementia_classifier(
        feature_extractor=extractor,
        classifier_head=qtl_head,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(qtl_head.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    trainer = BaselineTrainer(
        model=qtl_head,
        train_loader=train_cached_loader,
        test_loader=test_cached_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        save_dir=save_dir,
        save_model=full_model,
    )

    trainer.train(epochs=1, run_id="test_run_qtl")

    saved_checkpoint_path = Path(save_dir) / "baseline_test_run_qtl.pt"
    assert saved_checkpoint_path.exists()

    state_dict = torch.load(saved_checkpoint_path, weights_only=True)
    assert any(k.startswith("feature_extractor.") for k in state_dict.keys())
    assert any(k.startswith("classifier_head.") for k in state_dict.keys())

    evaluator_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=QuantumClassifierHead(
            in_features=FEATURE_DIM,
            n_qubits=TEST_QTL_QUBITS,
            n_layers=TEST_QTL_LAYERS,
        ),
    )
    evaluator = ModelEvaluator(model=evaluator_model, device=device)
    evaluator.load_weights(str(saved_checkpoint_path))

    y_true, y_prob = evaluator.predict(raw_loader)
    assert y_true.shape == (NUM_MOCK_SAMPLES,)
    assert y_prob.shape == (NUM_MOCK_SAMPLES,)
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()
