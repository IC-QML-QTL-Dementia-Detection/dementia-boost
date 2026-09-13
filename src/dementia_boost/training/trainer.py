"""Training loop orchestration, epoch scheduling, and model checkpointing.

This module provides `BaselineTrainer` to manage the complete training lifecycle
for dementia classification models, decoupling epoch loops, loss computation,
metric logging, test evaluation, and artifact saving from entry scripts.
"""

import os
from logging import Logger

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


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
        """
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

        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, epochs: int, run_id: str) -> None:
        """Executes the training loop across the requested number of epochs.

        Runs forward and backward passes, updates optimizer and scheduler states,
        logs per-epoch loss and accuracy metrics, and triggers evaluation and
        checkpointing upon training completion.

        Args:
            epochs: Total number of complete passes over the training dataset.
            run_id: Unique identifier for this run (e.g., "seed_42"), used for
                checkpoint file naming.
        """
        self.logger.info(f"Starting training run: {run_id} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
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

            self.scheduler.step()

            epoch_loss = running_loss / total_samples
            epoch_acc = correct_preds / total_samples

            self.logger.info(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_loss:.4f} "
                f"| Train Acc: {epoch_acc:.4f}",
            )

        self.logger.info(
            f"Training complete for run {run_id}. Starting final evaluation."
        )

        self._evaluate_and_save(run_id)

    def _evaluate_and_save(self, run_id: str) -> None:
        """Evaluates model performance on the test dataset and saves weights.

        Computes final test loss and accuracy in evaluation mode without gradients,
        logs test performance, and serializes the model state dictionary to disk.

        Args:
            run_id: Unique identifier for the run used in file naming.
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.float().view(-1, 1).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = (outputs >= self.LOGIT_CLASSIFICATION_THRESHOLD).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        final_loss = val_loss / val_total
        final_acc = val_correct / val_total

        self.logger.info(
            f"[*] Final Test Loss: {final_loss:.4f} | Final Test Acc: {final_acc:.4f}"
        )

        target_model = self.save_model if self.save_model is not None else self.model
        save_path = os.path.join(self.save_dir, f"baseline_{run_id}.pt")
        torch.save(target_model.state_dict(), save_path)
        self.logger.info(f"Final model saved to {save_path}")
