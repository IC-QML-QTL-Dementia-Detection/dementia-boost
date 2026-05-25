import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelEvaluator:
    """
    Handles trained model weights loading and the execution of
    inference over a dataset to extract raw probabilities and ground truths.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        Args:
            model (nn.Module): The uninitialized neural network architecture.
            device (torch.device): The target hardware accelerator.
        """
        self.model = model.to(device)
        self.device = device

    def load_weights(self, filepath: str) -> None:
        """
        Loads the state dictionary into the model from a .pt file.
        Maps tensors to the current device.

        Args:
            filepath (str): Path to the saved .pt file.
        """
        state_dict = torch.load(
            filepath,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the full dataset through the model and extracts predictions.

        Args:
            data_loader (DataLoader): The dataset to evaluate.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - y_true: A 1D numpy array of the actual ground-truth labels.
                - y_prob: A 1D numpy array of the model's raw probability outputs.
        """
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)

                probs = self.model(images).squeeze(dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_probs)
