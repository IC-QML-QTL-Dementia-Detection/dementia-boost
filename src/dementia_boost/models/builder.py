"""Model factory builders for Classical and Quantum Transfer Learning.

This module provides factory functions to load pre-trained classical baseline CNN
weights, freeze convolutional feature extraction layers, attach newly initialized
classical dense heads (CTL) or Dressed Quantum Network heads (QTL), extract baseline
backbones, and assemble complete DementiaClassifier models from separate components.
"""

import torch
import torch.nn as nn

from dementia_boost.models.quantum_cnn import QuantumClassifierHead

from .classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)


def load_baseline_backbone(
    baseline_weights_path: str,
    device: torch.device,
) -> LeNetFeatureExtractor:
    """Loads pre-trained baseline backbone weights and freezes parameters.

    Args:
        baseline_weights_path: Filepath to the serialized baseline checkpoint.
        device: Hardware accelerator device where backbone tensors reside.

    Returns:
        A frozen LeNetFeatureExtractor module allocated on device.
    """
    extractor = LeNetFeatureExtractor().to(device)
    temp_model = DementiaClassifier(
        feature_extractor=extractor,
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )

    state_dict = torch.load(
        baseline_weights_path,
        map_location=device,
        weights_only=True,
    )
    temp_model.load_state_dict(state_dict)

    for param in extractor.parameters():
        param.requires_grad = False

    return extractor


def build_classical_tl_model(
    baseline_weights_path: str,
    device: torch.device,
    use_sigmoid: bool = False,
) -> nn.Module:
    """Builds a Classical Transfer Learning (CTL) model.

    Loads a pre-trained baseline CNN checkpoint, freezes its convolutional
    backbone parameters, and resets the weights of its classical dense
    classification head using Glorot Uniform initialization.

    Args:
        baseline_weights_path: Filepath to the saved baseline `.pt` weight file.
        device: The target hardware accelerator device (CPU, CUDA, MPS).
        use_sigmoid: Whether to include a Sigmoid activation function in the
            classification head. Defaults to False.

    Returns:
        The prepared PyTorch module with frozen backbone and initialized head,
        allocated on the target device.
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
    n_qubits: int = QuantumClassifierHead.DEFAULT_N_QUBITS,
    n_layers: int = QuantumClassifierHead.DEFAULT_N_LAYERS,
) -> nn.Module:
    """Builds a Quantum Transfer Learning (QTL) hybrid model.

    Loads a pre-trained classical baseline CNN checkpoint, freezes its
    convolutional backbone parameters, and substitutes its classification head
    with a Dressed Quantum Network (pre-net + VQC + post-net) initialized
    with Glorot Uniform weights.

    Args:
        baseline_weights_path: Filepath to the saved baseline `.pt` weight file.
        device: The target hardware accelerator device (CPU, CUDA, MPS).
        n_qubits: Number of qubits in the variational quantum circuit.
        n_layers: Number of variational repetitions (depth) in the ansatz.

    Returns:
        The prepared hybrid PyTorch module with frozen backbone and initialized
        quantum head, allocated on the target device.
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
        in_features=QuantumClassifierHead.DEFAULT_IN_FEATURES,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    model.classifier_head.apply(QuantumClassifierHead.apply_glorot_init)
    return model.to(device)


def assemble_dementia_classifier(
    feature_extractor: nn.Module,
    classifier_head: nn.Module,
) -> DementiaClassifier:
    """Assembles a full DementiaClassifier orchestrator from components.

    Connects a trained or frozen spatial feature extraction backbone with a
    trained classical or quantum classification head into a single unified
    module suitable for end-to-end inference and checkpoint serialization.

    Args:
        feature_extractor: Feature extraction backbone module.
        classifier_head: Classification head module mapping features to logits.

    Returns:
        A composed DementiaClassifier instance.
    """
    return DementiaClassifier(
        feature_extractor=feature_extractor,
        classifier_head=classifier_head,
    )
