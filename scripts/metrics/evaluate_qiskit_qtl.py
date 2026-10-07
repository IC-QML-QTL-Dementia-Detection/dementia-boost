"""Batch evaluation engine for Qiskit Quantum Transfer Learning (QTL) models.

This script iterates across all saved Qiskit v2.x hybrid Dressed Quantum
Network (DQN) model checkpoints (.pt) trained on NIfTI axial slices, performs
batched inference on the validation and test cohorts, computes full binary
classification metrics (Accuracy, Precision, Recall, F1-score, AUC-ROC, log
loss, Confusion Matrix), aggregates statistical distributions, and exports one
telemetry JSON per cohort. Validation metrics are only for model selection; test
metrics are only reported.
"""

import os
import sys

import torch

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.checkpoint_evaluation import (
    evaluate_checkpoints,
    log_cohort_summaries,
    save_cohort_results,
)
from dementia_boost.training.evaluator import ModelEvaluator

DEFAULT_TORCH_DEVICE: str = "cpu"


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


def main() -> None:
    """Executes the batch evaluation pipeline for Qiskit hybrid QTL NIfTI models."""
    logger = setup_logger("evaluate_qiskit_qtl_nifti")
    device = get_device()

    models_dir = "./data/results/trained_qiskit_qtl_models/nifti"
    results_dir = "./data/results/metrics/nifti"

    n_qubits = 6
    n_layers = 4
    batch_size = 64

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(models_dir):
        logger.error(f"Checkpoint directory {models_dir} not found.")
        sys.exit(1)

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("baseline_qiskit_qtl_") and f.endswith(".pt")
    ]
    if not model_files:
        logger.error(f"No Qiskit QTL model checkpoint files found in {models_dir}.")
        sys.exit(1)

    model_files.sort(
        key=lambda f: (
            int(f.replace("baseline_qiskit_qtl_seed_", "").replace(".pt", ""))
            if f.replace("baseline_qiskit_qtl_seed_", "").replace(".pt", "").isdigit()
            else f
        )
    )

    base_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=QiskitQuantumClassifierHead(
            in_features=2304,
            n_qubits=n_qubits,
            n_layers=n_layers,
        ),
    )

    loader_manager = OasisDataLoader(batch_size=batch_size)
    loaders = {
        cohort: loader_manager.get_data_loader(cohort) for cohort in ("val", "test")
    }
    evaluator = ModelEvaluator(model=base_model, device=device)

    logger.info(f"Target PyTorch Device: {device}")
    logger.info(f"Found {len(model_files)} Qiskit QTL models. Beginning evaluation...")

    checkpoints = [
        (file_name.replace(".pt", ""), os.path.join(models_dir, file_name))
        for file_name in model_files
    ]
    results_by_cohort = evaluate_checkpoints(evaluator, checkpoints, loaders)
    paths = save_cohort_results(results_by_cohort, results_dir, "qiskit_qtl")

    logger.info(f"Success! Qiskit QTL evaluation complete. Results saved to {paths}")
    log_cohort_summaries(logger, results_by_cohort)


if __name__ == "__main__":
    main()
