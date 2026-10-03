"""Unit tests for BaselineTrainer lifecycle and checkpointing.

This module validates that BaselineTrainer supports both end-to-end full model
training on raw images and fast transfer learning on cached feature embeddings
with composite model checkpoint serialization for CTL and QTL heads, and that
it records a per-epoch `TrainingHistory` atomically to disk.
"""

import math
from pathlib import Path

import pytest
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
from dementia_boost.telemetry.metrics import MetricsAnalyzer
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
HISTORY_STEP_SIZE: int = 2
BATCHES_PER_EPOCH: int = NUM_MOCK_SAMPLES // BATCH_SIZE


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


class _FailingCriterion(nn.BCEWithLogitsLoss):
    """BCE loss that raises after a fixed number of calls to simulate a crash."""

    def __init__(self, max_calls: int) -> None:
        super().__init__()
        self.max_calls = max_calls
        self.calls = 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("simulated crash")
        return super().forward(input, target)


def _build_history_trainer(
    tmp_path: Path,
    criterion: nn.Module,
    eval_every: int,
) -> tuple[BaselineTrainer, Path]:
    """Builds a tiny linear-model trainer that records its history to disk."""
    features = torch.randn(NUM_MOCK_SAMPLES, 4)
    labels = torch.tensor([0.0, 1.0] * (NUM_MOCK_SAMPLES // 2))
    loader = DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE)
    model = nn.Linear(4, 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history_path = tmp_path / "histories" / "run.json"

    trainer = BaselineTrainer(
        model=model,
        train_loader=loader,
        test_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=HISTORY_STEP_SIZE, gamma=GAMMA),
        device=torch.device("cpu"),
        logger=setup_logger("test_trainer_history"),
        save_dir=str(tmp_path / "checkpoints"),
        history_path=str(history_path),
        eval_every=eval_every,
        paradigm="baseline",
    )
    return trainer, history_path


def test_train_returns_history_matching_schedule_and_disk(tmp_path: Path) -> None:
    """Validates epoch count, finite losses, the StepLR learning rate used in
    each epoch, validation only on `eval_every` boundaries, and that the JSON
    on disk equals the returned history."""
    trainer, history_path = _build_history_trainer(
        tmp_path, nn.BCEWithLogitsLoss(), eval_every=2
    )

    history = trainer.train(epochs=4, run_id="run")

    assert [r.epoch for r in history.epochs] == [1, 2, 3, 4]
    assert all(math.isfinite(r.train_loss) for r in history.epochs)
    assert [r.lr for r in history.epochs] == pytest.approx(
        [LEARNING_RATE, LEARNING_RATE, LEARNING_RATE * GAMMA, LEARNING_RATE * GAMMA]
    )
    assert [r.val_loss is not None for r in history.epochs] == [
        False,
        True,
        False,
        True,
    ]
    assert history.config["lr_step_size"] == HISTORY_STEP_SIZE
    assert MetricsAnalyzer.load_history(str(history_path)) == history


def test_history_survives_mid_run_crash(tmp_path: Path) -> None:
    """Validates that a crash in epoch 3 leaves a parseable JSON holding the two
    completed epochs and no leftover temporary file."""
    criterion = _FailingCriterion(max_calls=2 * BATCHES_PER_EPOCH)
    trainer, history_path = _build_history_trainer(tmp_path, criterion, eval_every=100)

    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer.train(epochs=4, run_id="run")

    saved = MetricsAnalyzer.load_history(str(history_path))
    assert [r.epoch for r in saved.epochs] == [1, 2]
    assert not history_path.with_name("run.json.tmp").exists()
