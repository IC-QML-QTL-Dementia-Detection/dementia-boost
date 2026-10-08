"""Training loop orchestration, epoch scheduling, and model checkpointing.

This module provides `BaselineTrainer` to manage the complete training lifecycle
for dementia classification models, decoupling epoch loops, loss computation,
metric logging, validation evaluation, and artifact saving from entry scripts.

The trainer only ever sees the train and validation loaders. The test cohort is
evaluated afterwards, from the saved checkpoints, so it can never influence
training or model selection.

The trainer takes its identity from a `RunSpec` and derives every artifact path
from a `ResultsLayout`; it refuses a spec that does not describe what runs, so
the hash IDs never name a configuration that did not run. It records a
structured per-epoch `TrainingHistory` carrying the spec, but never plots it;
visualization stays in the telemetry layer.
"""

import math
import os
import time
from logging import Logger
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from dementia_boost.core.identity import RunSpec, label
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)


class BaselineTrainer:
    """Manages the training, logging, and evaluation lifecycle of dementia models.

    Decouples the optimization loop, learning rate scheduling, real-time telemetry
    logging, and checkpointing from model definition and entry scripts.

    Attributes:
        LOGIT_CLASSIFICATION_THRESHOLD: Logit threshold for binary decision (0.0).
        model: The PyTorch neural network or hybrid module to be trained.
        train_loader: PyTorch DataLoader providing training mini-batches.
        val_loader: PyTorch DataLoader providing validation mini-batches.
        criterion: Loss function module computing objective loss.
        optimizer: Optimization algorithm updating model parameters.
        scheduler: Learning rate decay scheduler.
        device: Hardware accelerator device (CPU, CUDA, MPS) running the workload.
        logger: Telemetry logger streaming messages to console and disk.
        spec: The run specification: identity, hyperparameters, and epochs.
        layout: Where the run's checkpoint, history, and config are written.
        save_model: Optional PyTorch module whose state dictionary is saved to
            disk upon completion (e.g. an assembled DementiaClassifier).
        eval_every: Evaluate on `val_loader` every this many epochs. The last
            epoch is always evaluated, and that result is the final validation
            report.
        history_save_every: Write the history every this many epochs, and once
            more when training ends or fails.
        extras: Non-identity details stored in the history (not hashed).
    """

    LOGIT_CLASSIFICATION_THRESHOLD: float = 0.0

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: torch.device,
        logger: Logger,
        spec: RunSpec,
        layout: ResultsLayout,
        save_model: nn.Module | None = None,
        eval_every: int = 1,
        history_save_every: int = 10,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the BaselineTrainer with all required dependencies.

        Args:
            model: The neural network model to be trained.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation cohort, evaluated during
                training.
            criterion: The loss function module (e.g., BCEWithLogitsLoss).
            optimizer: The weight optimization algorithm (e.g., Adam).
            scheduler: The learning rate decay scheduler.
            device: The hardware accelerator device (CPU, CUDA, MPS).
            logger: The telemetry logger for console and file output.
            spec: The run specification. Its learning rate, batch size, and
                `StepLR` settings must match the optimizer, loader, and
                scheduler given here.
            layout: The results layout that decides every artifact path.
            save_model: Optional PyTorch module to serialize on disk instead of
                `model`. Defaults to None (saves `model`).
            eval_every: Number of epochs between evaluations on `val_loader`.
                Epochs in between record `None` validation values, except the
                last epoch, which is always evaluated. Defaults to 1.
            history_save_every: Number of epochs between history writes. The
                history stays in memory in between, and is always written once
                more when training ends or fails, so a crash loses nothing
                except on a hard kill (at most `history_save_every - 1` epochs).
                Defaults to 10.
            extras: Details that do not define the run, such as the quantum
                device. Stored in the history next to `eval_every` and
                `history_save_every`, and not part of the hash. Defaults to
                None.

        Raises:
            ValueError: If `eval_every` or `history_save_every` is below 1, or
                the spec disagrees with the optimizer, loader, or scheduler.
        """
        for name, value in (
            ("eval_every", eval_every),
            ("history_save_every", history_save_every),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.spec = spec
        self.layout = layout
        self.save_model = save_model
        self.eval_every = eval_every
        self.history_save_every = history_save_every
        self.extras = {
            "eval_every": eval_every,
            "history_save_every": history_save_every,
            **(extras or {}),
        }

        self._check_spec_matches_setup()

    def train(self) -> TrainingHistory:
        """Executes the training loop for `spec.epochs` epochs.

        Records the configuration next to the histories, runs forward and
        backward passes, updates optimizer and scheduler states, logs per-epoch
        loss and accuracy metrics, records a `TrainingHistory`, and saves the
        final checkpoint upon training completion. The validation loader is
        evaluated every `eval_every` epochs and always after the last epoch; the
        last evaluation doubles as the final validation report, so no extra pass
        runs at the end. The checkpoint is always the last epoch.

        The history is written every `history_save_every` epochs and once more
        on exit, including when training raises, so every completed epoch is on
        disk before the exception propagates.

        Returns:
            The completed TrainingHistory. Each epoch's `duration_s` covers the
            optimization pass only, excluding the optional evaluation.

        Raises:
            ConfigCollisionError: If a different spec already owns this
                configuration ID.
        """
        epochs = self.spec.epochs
        self.layout.write_config(self.spec)
        self.logger.info(
            f"Starting training run: {label(self.spec)} for {epochs} epochs."
        )

        history = TrainingHistory(spec=self.spec, epochs=[], extras=self.extras)
        saved_epochs = 0
        last_evaluation: tuple[float, float] | None = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                epoch_loss, epoch_acc = self._train_one_epoch()

                epoch_lr = float(self.scheduler.get_last_lr()[0])
                self.scheduler.step()
                epoch_duration = time.perf_counter() - epoch_start

                self.logger.info(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: "
                    f"{epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}",
                )

                val_loss, val_acc = None, None
                if epoch % self.eval_every == 0 or epoch == epochs:
                    last_evaluation = self._evaluate_loader(self.val_loader)
                    val_loss, val_acc = last_evaluation

                history.epochs.append(
                    EpochRecord(
                        epoch=epoch,
                        train_loss=epoch_loss,
                        train_acc=epoch_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        lr=epoch_lr,
                        duration_s=epoch_duration,
                    )
                )

                if epoch % self.history_save_every == 0:
                    self._save_history(history)
                    saved_epochs = len(history.epochs)
        finally:
            if len(history.epochs) > saved_epochs:
                self._save_history(history)

        self.logger.info(
            f"Training complete for run {label(self.spec)}. Saving final model."
        )

        if last_evaluation is None:
            last_evaluation = self._evaluate_loader(self.val_loader)
        self._save_checkpoint(*last_evaluation)

        return history

    def _train_one_epoch(self) -> tuple[float, float]:
        """Runs one optimization pass over the training loader.

        Returns:
            A `(loss, accuracy)` tuple averaged over all training samples.
        """
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().view(-1, 1).to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = (outputs >= self.LOGIT_CLASSIFICATION_THRESHOLD).float()
            correct_preds += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        return running_loss / total_samples, correct_preds / total_samples

    def _check_spec_matches_setup(self) -> None:
        """Checks that the spec describes the optimizer, loader, and scheduler.

        The IDs are hashes of the spec, so a spec that says one thing while the
        run does another would name a configuration that never ran.

        Raises:
            ValueError: Naming every field of the spec that disagrees.
        """
        mismatches: list[str] = []

        actual_lr = float(self.scheduler.get_last_lr()[0])
        if not math.isclose(self.spec.lr, actual_lr):
            mismatches.append(f"lr: spec {self.spec.lr}, optimizer {actual_lr}")
        if self.spec.batch_size != self.train_loader.batch_size:
            mismatches.append(
                f"batch_size: spec {self.spec.batch_size}, "
                f"loader {self.train_loader.batch_size}"
            )
        if isinstance(self.scheduler, StepLR):
            if self.spec.lr_step_size != self.scheduler.step_size:
                mismatches.append(
                    f"lr_step_size: spec {self.spec.lr_step_size}, "
                    f"scheduler {self.scheduler.step_size}"
                )
            if not math.isclose(self.spec.lr_gamma, self.scheduler.gamma):
                mismatches.append(
                    f"lr_gamma: spec {self.spec.lr_gamma}, "
                    f"scheduler {self.scheduler.gamma}"
                )

        if mismatches:
            raise ValueError(
                "The run spec disagrees with what will run: " + "; ".join(mismatches)
            )

    def _save_history(self, history: TrainingHistory) -> None:
        """Atomically writes the running history to the layout's history path.

        Writes to a sibling `.tmp` file and swaps it in with `os.replace`, so a
        crash never leaves a partially written JSON.

        Args:
            history: The running TrainingHistory to persist.
        """
        path = self.layout.history_path(self.spec)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        MetricsAnalyzer.save_history(history, tmp_path)
        os.replace(tmp_path, path)

    def _evaluate_loader(self, loader: DataLoader) -> tuple[float, float]:
        """Computes mean loss and accuracy over a loader without gradients.

        Args:
            loader: DataLoader yielding `(inputs, labels)` batches.

        Returns:
            A `(loss, accuracy)` tuple averaged over all samples in `loader`.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().view(-1, 1).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                predictions = (outputs >= self.LOGIT_CLASSIFICATION_THRESHOLD).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, final_loss: float, final_acc: float) -> None:
        """Logs the final validation performance and saves the model weights.

        Receives the validation loss and accuracy of the last epoch instead of
        evaluating again, since the weights have not changed since then.
        Serializes the model state dictionary to the layout's checkpoint path.

        Args:
            final_loss: Validation loss measured after the last epoch.
            final_acc: Validation accuracy measured after the last epoch.
        """
        self.logger.info(
            f"[*] Final Val Loss: {final_loss:.4f} | Final Val Acc: {final_acc:.4f}"
        )

        target_model = self.save_model if self.save_model is not None else self.model
        save_path = self.layout.checkpoint_path(self.spec)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(target_model.state_dict(), save_path)
        self.logger.info(f"Final model saved to {save_path}")
