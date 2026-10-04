"""Training loop orchestration, epoch scheduling, and model checkpointing.

This module provides `BaselineTrainer` to manage the complete training lifecycle
for dementia classification models, decoupling epoch loops, loss computation,
metric logging, test evaluation, and artifact saving from entry scripts.

The trainer records a structured per-epoch `TrainingHistory` but never plots it;
visualization stays in the telemetry layer.
"""

import os
import time
from logging import Logger
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)


class BaselineTrainer:
    """Manages the training, logging, and evaluation lifecycle of dementia models.

    Decouples the optimization loop, learning rate scheduling, real-time telemetry
    logging, and test set checkpointing from model definition and entry scripts.

    Attributes:
        DEFAULT_SAVE_DIR: Default directory where model weights are written.
        LOGIT_CLASSIFICATION_THRESHOLD: Logit threshold for binary decision (0.0).
        model: The PyTorch neural network or hybrid module to be trained.
        train_loader: PyTorch DataLoader providing training mini-batches.
        test_loader: PyTorch DataLoader providing test/validation mini-batches.
        criterion: Loss function module computing objective loss.
        optimizer: Optimization algorithm updating model parameters.
        scheduler: Learning rate decay scheduler.
        device: Hardware accelerator device (CPU, CUDA, MPS) running the workload.
        logger: Telemetry logger streaming messages to console and disk.
        save_dir: Directory where checkpoint `.pt` files are written.
        save_model: Optional PyTorch module whose state dictionary is saved to
            disk upon completion (e.g. an assembled DementiaClassifier).
        history_path: Optional JSON file holding the running `TrainingHistory`,
            rewritten atomically. None disables persistence.
        eval_every: Evaluate on `test_loader` every this many epochs. The last
            epoch is always evaluated, and that result is the final test report.
        history_save_every: Write the history to `history_path` every this many
            epochs, and once more when training ends or fails.
        paradigm: Training paradigm label stored in the history.
        config: Extra hyperparameters stored in the history config.
    """

    DEFAULT_SAVE_DIR: str = "./data/results/trained_models"
    LOGIT_CLASSIFICATION_THRESHOLD: float = 0.0

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: torch.device,
        logger: Logger,
        save_dir: str = DEFAULT_SAVE_DIR,
        save_model: nn.Module | None = None,
        history_path: str | None = None,
        eval_every: int = 1,
        history_save_every: int = 10,
        paradigm: str = "baseline",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the BaselineTrainer with all required dependencies.

        Args:
            model: The neural network model to be trained.
            train_loader: DataLoader for the training dataset.
            test_loader: DataLoader for the final testing dataset.
            criterion: The loss function module (e.g., BCEWithLogitsLoss).
            optimizer: The weight optimization algorithm (e.g., Adam).
            scheduler: The learning rate decay scheduler.
            device: The hardware accelerator device (CPU, CUDA, MPS).
            logger: The telemetry logger for console and file output.
            save_dir: Directory where final model weights will be saved.
                Defaults to "./data/results/trained_models".
            save_model: Optional PyTorch module to serialize on disk instead of
                `model`. Defaults to None (saves `model`).
            history_path: Optional JSON path for the per-epoch history, written
                atomically. Defaults to None (no history file).
            eval_every: Number of epochs between evaluations on `test_loader`.
                Epochs in between record `None` validation values, except the
                last epoch, which is always evaluated. Defaults to 1.
            history_save_every: Number of epochs between history writes. The
                history stays in memory in between, and is always written once
                more when training ends or fails, so a crash loses nothing
                except on a hard kill (at most `history_save_every - 1` epochs).
                Defaults to 10.
            paradigm: Training paradigm label ("baseline", "ctl", "qtl" or
                "qiskit_qtl"). Defaults to "baseline".
            config: Extra hyperparameters (for example `n_qubits`, `n_layers`)
                merged over the values the trainer derives itself. Defaults to
                None.

        Raises:
            ValueError: If `eval_every` or `history_save_every` is below 1.
        """
        for name, value in (
            ("eval_every", eval_every),
            ("history_save_every", history_save_every),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.save_model = save_model
        self.history_path = history_path
        self.eval_every = eval_every
        self.history_save_every = history_save_every
        self.paradigm = paradigm
        self.config = config or {}

        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, epochs: int, run_id: str) -> TrainingHistory:
        """Executes the training loop across the requested number of epochs.

        Runs forward and backward passes, updates optimizer and scheduler states,
        logs per-epoch loss and accuracy metrics, records a `TrainingHistory`,
        and saves the final checkpoint upon training completion. The test loader
        is evaluated every `eval_every` epochs and always after the last epoch;
        the last evaluation doubles as the final test report, so no extra pass
        runs at the end.

        The history is written to `history_path` every `history_save_every`
        epochs and once more on exit, including when training raises, so every
        completed epoch is on disk before the exception propagates.

        Args:
            epochs: Total number of complete passes over the training dataset.
            run_id: Unique identifier for this run (e.g., "seed_42"), used for
                checkpoint file naming and as the history run identifier.

        Returns:
            The completed TrainingHistory. Each epoch's `duration_s` covers the
            optimization pass only, excluding the optional evaluation.
        """
        self.logger.info(f"Starting training run: {run_id} for {epochs} epochs.")

        history = TrainingHistory(
            run_id=run_id,
            paradigm=self.paradigm,
            config=self._build_history_config(epochs),
            epochs=[],
        )
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
                    last_evaluation = self._evaluate_loader(self.test_loader)
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

        self.logger.info(f"Training complete for run {run_id}. Saving final model.")

        if last_evaluation is None:
            last_evaluation = self._evaluate_loader(self.test_loader)
        self._save_checkpoint(run_id, *last_evaluation)

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

    def _build_history_config(self, epochs: int) -> dict[str, Any]:
        """Builds the reproducibility config stored in the training history.

        Args:
            epochs: Total number of epochs requested for the run.

        Returns:
            A dictionary with the learning rate, batch size, epochs, evaluation
            cadence, and the `StepLR` step size when applicable, updated with
            the user-supplied `config` entries.
        """
        derived: dict[str, Any] = {
            "lr": float(self.scheduler.get_last_lr()[0]),
            "batch_size": self.train_loader.batch_size,
            "epochs": epochs,
            "eval_every": self.eval_every,
        }
        if isinstance(self.scheduler, StepLR):
            derived["lr_step_size"] = self.scheduler.step_size
            derived["lr_gamma"] = self.scheduler.gamma

        return {**derived, **self.config}

    def _save_history(self, history: TrainingHistory) -> None:
        """Atomically writes the running history to `history_path`.

        Writes to a sibling `.tmp` file and swaps it in with `os.replace`, so a
        crash never leaves a partially written JSON. Does nothing when
        `history_path` is None.

        Args:
            history: The running TrainingHistory to persist.
        """
        if self.history_path is None:
            return

        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        tmp_path = f"{self.history_path}.tmp"
        MetricsAnalyzer.save_history(history, tmp_path)
        os.replace(tmp_path, self.history_path)

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

    def _save_checkpoint(
        self, run_id: str, final_loss: float, final_acc: float
    ) -> None:
        """Logs the final test performance and saves the model weights.

        Receives the test loss and accuracy of the last epoch instead of
        evaluating again, since the weights have not changed since then.
        Serializes the model state dictionary to disk.

        Args:
            run_id: Unique identifier for the run used in file naming.
            final_loss: Test loss measured after the last epoch.
            final_acc: Test accuracy measured after the last epoch.
        """
        self.logger.info(
            f"[*] Final Test Loss: {final_loss:.4f} | Final Test Acc: {final_acc:.4f}"
        )

        target_model = self.save_model if self.save_model is not None else self.model
        save_path = os.path.join(self.save_dir, f"baseline_{run_id}.pt")
        torch.save(target_model.state_dict(), save_path)
        self.logger.info(f"Final model saved to {save_path}")
