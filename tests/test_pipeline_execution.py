"""Unit tests for the training loop lifecycle and inference evaluator.

This module validates:
- ``BaselineTrainer`` end-to-end raw-image training loss reduction, checkpoint
  key structure, and optimizer state advancement.
- ``BaselineTrainer`` learning rate scheduler stepping behavior.
- ``ModelEvaluator`` batched inference probability bounds and eval-mode
  invariance.
- ``ModelEvaluator`` checkpoint weight restoration exactness.

Regression coverage
-------------------
- Training loops that silently fail to update parameters, leaving loss flat.
- Checkpoints missing the composite `feature_extractor.*` / `classifier_head.*`
  key structure required by downstream loaders.
- Learning rate schedulers that drift from the configured step/gamma contract.
- Evaluators that leave the model in training mode, activating dropout/batchnorm
  noise during inference.
- Weight loading that fails to exactly restore serialized parameters.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.evaluator import ModelEvaluator
from dementia_boost.training.trainer import BaselineTrainer

_NUM_MOCK_SAMPLES: int = 8
_IMAGE_SIZE: int = 128
_BATCH_SIZE: int = 4
_SEED: int = 42
_MOCK_LABELS: torch.Tensor = torch.tensor([0.0, 1.0] * (_NUM_MOCK_SAMPLES // 2))


def _build_mock_image_loader() -> DataLoader:
    """Builds a small deterministic DataLoader of random MRI-shaped tensors.

    Returns:
        A DataLoader yielding `(image, label)` batches of shape
        `(Batch, 1, 128, 128)` and `(Batch,)` with alternating binary labels.
    """
    images = torch.randn(_NUM_MOCK_SAMPLES, 1, _IMAGE_SIZE, _IMAGE_SIZE)
    dataset = TensorDataset(images, _MOCK_LABELS)
    return DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=False)


def _build_end_to_end_model() -> DementiaClassifier:
    """Builds a fully trainable classical `DementiaClassifier` for E2E tests.

    Disables dropout so that repeated forward passes on the same input are
    deterministic in eval mode, isolating loss changes to weight updates.

    Returns:
        A `DementiaClassifier` combining a fresh `LeNetFeatureExtractor` and
        `ClassicalClassifierHead`.
    """
    return DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False, dropout_rate=0.0),
    )


class TestBaselineTrainerEndToEnd:
    """Validates full end-to-end training directly on raw image tensors."""

    def test_raw_image_training_reduces_loss_and_checkpoints(
        self,
        tmp_path: Path,
    ) -> None:
        """Runs multiple epochs of end-to-end CNN training on raw images.

        Asserts that the loss on a fixed evaluation batch decreases after
        training, that the checkpoint is saved with the composite
        `feature_extractor.*` / `classifier_head.*` key structure, and that
        the optimizer accumulated per-parameter state (i.e. stepped).
        """
        set_seed(_SEED)
        device = torch.device("cpu")
        logger = setup_logger("test_pipeline_execution_e2e")
        save_dir = str(tmp_path / "checkpoints")

        model = _build_end_to_end_model().to(device)
        train_loader = _build_mock_image_loader()
        test_loader = _build_mock_image_loader()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        fixed_images, fixed_labels = next(iter(train_loader))
        fixed_labels = fixed_labels.float().view(-1, 1)

        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(fixed_images), fixed_labels).item()

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=save_dir,
        )
        trainer.train(epochs=5, run_id="e2e_test")

        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(fixed_images), fixed_labels).item()

        assert final_loss < initial_loss
        assert len(optimizer.state_dict()["state"]) > 0

        checkpoint_path = Path(save_dir) / "baseline_e2e_test.pt"
        assert checkpoint_path.exists()

        state_dict = torch.load(checkpoint_path, weights_only=True)
        assert any(k.startswith("feature_extractor.") for k in state_dict)
        assert any(k.startswith("classifier_head.") for k in state_dict)


class TestBaselineTrainerLrScheduler:
    """Validates learning rate decay contracts during training."""

    def test_lr_reduces_by_gamma_after_step_size_epochs(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that `StepLR` reduces the learning rate by `gamma` exactly
        once after `step_size` completed epochs."""
        device = torch.device("cpu")
        logger = setup_logger("test_pipeline_execution_lr")
        save_dir = str(tmp_path / "checkpoints_lr")

        model = _build_end_to_end_model().to(device)
        train_loader = _build_mock_image_loader()
        test_loader = _build_mock_image_loader()

        initial_lr = 1e-2
        step_size = 2
        gamma = 0.1

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=save_dir,
        )
        trainer.train(epochs=step_size, run_id="lr_test")

        lr_after_step = optimizer.param_groups[0]["lr"]
        assert lr_after_step == pytest.approx(initial_lr * gamma)


class TestModelEvaluatorPredict:
    """Validates batched inference contracts of `ModelEvaluator.predict`."""

    def test_predict_returns_bounded_probabilities_and_matching_labels(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that `y_true` matches dataset labels, `y_prob` values lie
        strictly in [0.0, 1.0], and the model remains in eval mode."""
        device = torch.device("cpu")
        model = _build_end_to_end_model()
        checkpoint_path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), checkpoint_path)

        evaluator = ModelEvaluator(model=model, device=device)
        evaluator.load_weights(str(checkpoint_path))

        loader = _build_mock_image_loader()
        y_true, y_prob = evaluator.predict(loader)

        np.testing.assert_array_equal(y_true, _MOCK_LABELS.numpy())
        assert (y_prob >= 0.0).all()
        assert (y_prob <= 1.0).all()
        assert evaluator.model.training is False


class TestModelEvaluatorLoadWeights:
    """Validates exact checkpoint weight restoration."""

    def test_load_weights_restores_exact_parameters(self, tmp_path: Path) -> None:
        """Saves known weights, perturbs a separate model, reloads the saved
        weights via the evaluator, and asserts exact parameter equality."""
        device = torch.device("cpu")
        source_model = _build_end_to_end_model()
        checkpoint_path = tmp_path / "weights.pt"
        torch.save(source_model.state_dict(), checkpoint_path)

        target_model = _build_end_to_end_model()
        with torch.no_grad():
            for param in target_model.parameters():
                param.add_(1.0)

        evaluator = ModelEvaluator(model=target_model, device=device)
        evaluator.load_weights(str(checkpoint_path))

        for source_param, loaded_param in zip(
            source_model.parameters(),
            evaluator.model.parameters(),
            strict=True,
        ):
            assert torch.equal(source_param, loaded_param)
