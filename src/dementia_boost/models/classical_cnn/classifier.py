import torch.nn as nn
from torch import Tensor


class DementiaClassifier(nn.Module):
    """
    The orchestrator module that composes a feature extractor and a classifier head.

    This architecture uses Dependency Injection to strictly separate the spatial
    representation logic (the backbone) from the decision-making logic (the head),
    allowing heads to be hot-swapped.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier_head: nn.Module,
    ) -> None:
        """
        Initializes the composed classifier model.

        Args:
            feature_extractor (nn.Module): The network backbone responsible for
                extracting spatial features from the raw image.
            classifier_head (nn.Module): The classification network responsible
                for mapping the extracted features to a binary prediction.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the full forward pass through the entire network.

        Args:
            x (Tensor): The raw input image tensor.

        Returns:
            Tensor: The final network prediction.
        """
        features = self.feature_extractor(x)
        prediction = self.classifier_head(features)

        return prediction
