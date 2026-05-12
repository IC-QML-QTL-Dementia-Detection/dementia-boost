import os
from logging import Logger

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


class BaselineTrainer:
    """
    Manages the training lifecycle for the classical Dementia CNN.

    This class decouples the training loop, metric tracking, and file saving
    from the model definitions and the entry-point scripts. It executes a
    pure training phase and evaluates/saves the model only upon completion.
    """

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
        save_dir: str = "./data/results/trained_models",
    ) -> None:
        """
        Initializes the BaselineTrainer with all required dependencies.

        Args:
            model (nn.Module): The classical CNN model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the final testing dataset.
            criterion (nn.Module): The loss function (e.g., BCELoss).
            optimizer (Optimizer): The weight optimization algorithm (e.g., Adam).
            scheduler (LRScheduler): The learning rate decay scheduler.
            device (torch.device): The hardware accelerator (CPU/CUDA/MPS).
            logger (Logger): The telemetry logger for console and file output.
            save_dir (str): Directory where the final model weights will be saved.
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

        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, epochs: int, run_id: str) -> None:
        """
        Executes the training loop for the specified number of epochs.

        Once all epochs are complete, it evaluates the model against the test
        set and saves the final weights to disk.

        Args:
            epochs (int): The total number of passes over the training dataset.
            run_id (str): A unique identifier for this run (e.g., "seed_42").
                Used for naming the saved model file.
        """
        self.logger.info(f"Starting training run: {run_id} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                predictions = outputs.round()
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
        """
        Evaluates the trained model on the test dataset and saves the weights.

        Args:
            run_id (str): The unique identifier for file naming.
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.round()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        final_loss = val_loss / val_total
        final_acc = val_correct / val_total

        self.logger.info(
            f"[*] Final Test Loss: {final_loss:.4f} | Final Test Acc: {final_acc:.4f}"
        )

        save_path = os.path.join(self.save_dir, f"baseline_{run_id}.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Final model saved to {save_path}")
