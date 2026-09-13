"""Batch evaluation engine for Quantum Transfer Learning (QTL) models on NIfTI data.

This script iterates across all saved hybrid Dressed Quantum Network (DQN)
model checkpoints (.pt) trained on NIfTI axial slices, performs batched inference
on the isolated test dataset, computes full binary classification metrics (Accuracy,
Precision, Recall, F1-score, AUC-ROC, Confusion Matrix), aggregates statistical
distributions, and exports the telemetry results to JSON.
"""

import os
import sys

import torch

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import QuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.metrics import MetricsAnalyzer
from dementia_boost.training.evaluator import ModelEvaluator

DEFAULT_TORCH_DEVICE: str = "cpu"
DEFAULT_QUANTUM_DEVICE: str = "lightning.qubit"


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
    """Executes the batch evaluation pipeline for hybrid QTL NIfTI models."""
    logger = setup_logger("evaluate_qtl_nifti")
    device = get_device()

    models_dir = "./data/results/trained_qtl_models/nifti"
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
        if f.startswith("baseline_qtl_") and f.endswith(".pt")
    ]
    if not model_files:
        logger.error(f"No QTL model checkpoint files found in {models_dir}.")
        sys.exit(1)

    model_files.sort(
        key=lambda f: (
            int(f.replace("baseline_qtl_seed_", "").replace(".pt", ""))
            if f.replace("baseline_qtl_seed_", "").replace(".pt", "").isdigit()
            else f
        )
    )

    base_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=QuantumClassifierHead(
            in_features=2304,
            n_qubits=n_qubits,
            n_layers=n_layers,
            quantum_device=DEFAULT_QUANTUM_DEVICE,
        ),
    )

    loader_manager = OasisDataLoader(batch_size=batch_size, mode="nifti")
    test_loader = loader_manager.get_data_loader(is_train=False)
    evaluator = ModelEvaluator(model=base_model, device=device)

    all_results = []
    logger.info(f"Target PyTorch Device: {device}")
    logger.info(f"Target Quantum Device: {DEFAULT_QUANTUM_DEVICE}")
    logger.info(f"Found {len(model_files)} QTL models. Beginning evaluation...")

    for file_name in model_files:
        run_id = file_name.replace(".pt", "")
        file_path = os.path.join(models_dir, file_name)

        logger.info(f"Evaluating QTL checkpoint: {run_id}...")
        evaluator.load_weights(file_path)

        y_true, y_prob = evaluator.predict(test_loader)

        result_dto = MetricsAnalyzer.calculate_metrics(run_id, y_true, y_prob)
        all_results.append(result_dto)

    logger.info("Aggregating statistical metrics across all QTL runs...")
    aggregated_stats = MetricsAnalyzer.aggregate_results(all_results)

    output_json = os.path.join(results_dir, "qtl_results.json")
    MetricsAnalyzer.save_to_json(all_results, aggregated_stats, output_json)

    logger.info(f"Success! QTL evaluation complete. Results saved to {output_json}")

    logger.info(
        f"Mean Acc: {aggregated_stats['accuracy'].mean:.4f} "
        f"\\pm {aggregated_stats['accuracy'].std:.4f}"
    )
    logger.info(
        f"Mean Precision: {aggregated_stats['precision'].mean:.4f} "
        f"\\pm {aggregated_stats['precision'].std:.4f}"
    )
    logger.info(
        f"Mean Recall: {aggregated_stats['recall'].mean:.4f} "
        f"\\pm {aggregated_stats['recall'].std:.4f}"
    )
    logger.info(
        f"Mean F1: {aggregated_stats['f1_score'].mean:.4f} "
        f"\\pm {aggregated_stats['f1_score'].std:.4f}"
    )
    logger.info(
        f"Mean AUC: {aggregated_stats['auc'].mean:.4f} "
        f"\\pm {aggregated_stats['auc'].std:.4f}"
    )


if __name__ == "__main__":
    main()
