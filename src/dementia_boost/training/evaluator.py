"""Model weight loading and batched inference evaluation engine.

This module provides `ModelEvaluator` to load trained checkpoint weights onto
target hardware devices and execute batched inference over PyTorch DataLoaders
to extract raw prediction probabilities and ground-truth labels.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelEvaluator:
    """Manages model checkpoint loading and batched dataset inference.

    Attributes:
        model: The PyTorch neural network or hybrid module to evaluate.
        device: The hardware accelerator device (CPU, CUDA, MPS) where tensors
            and model weights reside.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initializes the evaluator with the given model and execution device.

        Args:
            model: The neural network or hybrid architecture to evaluate.
            device: The target hardware accelerator device.
        """
        self.model = model.to(device)
        self.device = device

    def load_weights(self, filepath: str) -> None:
        """Loads model state dictionary weights from a serialized `.pt` checkpoint.

        Maps tensors directly to `self.device` and puts the model into
        evaluation mode (`self.model.eval()`).

        Args:
            filepath: Path to the saved `.pt` checkpoint file on disk.
        """
        state_dict = torch.load(
            filepath,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Executes batched inference over a DataLoader without gradient tracking.

        Passes all batches through the model in evaluation mode, applies a
        sigmoid activation to convert raw logits to probabilities, and compiles
        the ground-truth labels and predictions into NumPy arrays.

        Args:
            data_loader: The PyTorch DataLoader containing the dataset to evaluate.

        Returns:
            A tuple containing:
                - `y_true`: 1D NumPy array of ground-truth target labels.
                - `y_prob`: 1D NumPy array of raw model prediction probabilities.
        """
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)

                logits = self.model(images).squeeze(dim=1)
                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_probs)
