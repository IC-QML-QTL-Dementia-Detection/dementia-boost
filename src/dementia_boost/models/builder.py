import torch
import torch.nn as nn

from .classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)


def build_classical_tl_model(
    baseline_weights_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Builds a Classical Transfer Learning model by loading a pre-trained baseline,
    freezing its convolutional backbone, and resetting its dense head.

    Args:
        baseline_weights_path (str): Filepath to the saved baseline .pt file.
        device (torch.device): The target hardware accelerator.

    Returns:
        nn.Module: The prepared model, ready for fine-tuning.
    """

    model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(),
    )

    state_dict = torch.load(
        baseline_weights_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    model.classifier_head.apply(ClassicalClassifierHead.apply_glorot_init)

    return model.to(device)
