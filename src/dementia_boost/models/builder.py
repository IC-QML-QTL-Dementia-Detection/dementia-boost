import torch
import torch.nn as nn

from dementia_boost.models.quantum_cnn import QuantumClassifierHead

from .classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)


def build_classical_tl_model(
    baseline_weights_path: str,
    device: torch.device,
    use_sigmoid: bool = False,
) -> nn.Module:
    """
    Builds a Classical Transfer Learning model by loading a pre-trained baseline,
    freezing its convolutional backbone, and resetting its dense head.

    Args:
        baseline_weights_path (str): Filepath to the saved baseline .pt file.
        device (torch.device): The target hardware accelerator.
        use_sigmoid (bool): Whether there should be a Sigmoid activation function
            or not.

    Returns:
        nn.Module: The prepared model, ready for fine-tuning.
    """

    model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=use_sigmoid),
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


def build_quantum_tl_model(
    baseline_weights_path: str,
    device: torch.device,
    n_qubits: int,
    n_layers: int,
) -> nn.Module:
    """
    Builds a Quantum TL model by loading a pre-trained classical baseline, freezing
    its convolutional backbone, and replacing its dense head with a Dressed QN.

    Args:
        baseline_weights_path (str): Filepath to the saved baseline .pt file.
        device (torch.device): The target hardware accelerator.
        n_qubits (int): Number of qubits in the VQC. Defaults to 6.
        n_layers (int): Number of repetitions in the ansatz. Defaults to 4.

    Returns:
        nn.Module: The prepared model, ready for fine-tuning.
    """
    model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )

    state_dict = torch.load(
        baseline_weights_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    model.classifier_head = QuantumClassifierHead(
        in_features=2304,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    model.classifier_head.apply(QuantumClassifierHead.apply_glorot_init)
    return model.to(device)
