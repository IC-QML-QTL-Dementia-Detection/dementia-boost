"""Batch evaluation engine for Qiskit Quantum Transfer Learning (QTL) models.

This script finds every finished Qiskit v2.x hybrid Dressed Quantum Network (DQN)
run through its training history, performs batched inference on the validation
and test cohorts for each configuration (building each architecture from the
qubits and layers recorded in its spec), computes full binary classification
metrics (Accuracy, Precision, Recall, F1-score, AUC-ROC, log loss, Confusion
Matrix), aggregates statistical distributions, and exports one telemetry JSON per
cohort. Validation metrics are only for model selection; test metrics are only
reported.
"""

import sys

import torch

from dementia_boost.core.identity import Paradigm, RunSpec
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.checkpoint_evaluation import evaluate_paradigm

DEFAULT_TORCH_DEVICE: str = "cpu"
DEFAULT_FEATURE_DIM: int = 2304
DEFAULT_BATCH_SIZE: int = 64


def get_device(device_name: str | None = None) -> torch.device:
    """Resolves the target PyTorch execution device with optional manual override.

    When an explicit device string is provided, returns that device. If no
    override is given, defaults to `DEFAULT_TORCH_DEVICE` to avoid unnecessary GPU
    transfer latency for low-qubit quantum transfer learning workflows, while
    supporting 'auto' for automatic accelerator detection.

    Args:
        device_name: Optional device string ('cpu', 'cuda', 'mps', 'auto').
            If None, uses `DEFAULT_TORCH_DEVICE`. If 'auto', detects available
            hardware accelerators (CUDA, MPS) with fallback to CPU.

    Returns:
        A torch.device instance.
    """
    target = device_name if device_name is not None else DEFAULT_TORCH_DEVICE

    if target == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(target)


def build_model(spec: RunSpec) -> DementiaClassifier:
    """Builds the Qiskit QTL architecture a configuration's checkpoints fit.

    Args:
        spec: Spec of one run of the configuration; its qubits and layers set the
            size of the quantum head.

    Returns:
        An untrained classifier with a Qiskit quantum head.
    """
    if spec.n_qubits is None or spec.n_layers is None:
        raise ValueError("A Qiskit QTL spec must define n_qubits and n_layers.")
    return DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=QiskitQuantumClassifierHead(
            in_features=DEFAULT_FEATURE_DIM,
            n_qubits=spec.n_qubits,
            n_layers=spec.n_layers,
        ),
    )


def main() -> None:
    """Executes the batch evaluation pipeline for Qiskit hybrid QTL NIfTI models."""
    logger = setup_logger("evaluate_qiskit_qtl_nifti")
    device = get_device()
    logger.info(f"Target PyTorch Device: {device}")

    loader_manager = OasisDataLoader(batch_size=DEFAULT_BATCH_SIZE)
    loaders = {
        cohort: loader_manager.get_data_loader(cohort) for cohort in ("val", "test")
    }

    outcome = evaluate_paradigm(
        ResultsLayout(), Paradigm.QISKIT_QTL, build_model, loaders, device, logger
    )
    if not outcome:
        logger.error(
            "No finished Qiskit QTL runs found. Run its training script first."
        )
        sys.exit(1)

    logger.info(
        f"Success! Qiskit QTL evaluation complete for {len(outcome)} configuration(s)."
    )


if __name__ == "__main__":
    main()
