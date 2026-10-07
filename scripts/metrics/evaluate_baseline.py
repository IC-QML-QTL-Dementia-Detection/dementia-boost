"""Batch evaluation engine for classical baseline CNN models on NIfTI data.

This script iterates across all saved classical baseline checkpoints (.pt)
trained on NIfTI MRI axial slices, executes inference on the validation and test
cohorts, computes comprehensive binary classification metrics (Accuracy,
Precision, Recall, F1-score, AUC-ROC, log loss, Confusion Matrix), aggregates
statistical distributions across runs, and serializes one telemetry payload per
cohort to JSON. Validation metrics are only for model selection (the transfer
learning scripts read them to pick a backbone); test metrics are only reported.
"""

import os
import sys

import torch

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.checkpoint_evaluation import (
    evaluate_checkpoints,
    log_cohort_summaries,
    save_cohort_results,
)
from dementia_boost.training.evaluator import ModelEvaluator


def get_device() -> torch.device:
    """Selects the best available hardware accelerator device.

    Returns:
        A torch.device corresponding to CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Executes the batch evaluation pipeline for classical NIfTI baseline models."""
    logger = setup_logger("evaluate_baseline_nifti")
    device = get_device()
    models_dir = "./data/results/trained_models/nifti"
    results_dir = "./data/results/metrics/nifti"

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(models_dir):
        logger.error(f"Checkpoint directory {models_dir} not found.")
        sys.exit(1)

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])
    if not model_files:
        logger.error(f"No .pt model checkpoint files found in {models_dir}.")
        sys.exit(1)

    base_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )

    batch_size = 64
    loader_manager = OasisDataLoader(batch_size=batch_size)
    loaders = {
        cohort: loader_manager.get_data_loader(cohort) for cohort in ("val", "test")
    }
    evaluator = ModelEvaluator(model=base_model, device=device)

    logger.info(f"Found {len(model_files)} models. Beginning batch evaluation...")

    checkpoints = [
        (file_name.replace(".pt", ""), os.path.join(models_dir, file_name))
        for file_name in model_files
    ]
    results_by_cohort = evaluate_checkpoints(evaluator, checkpoints, loaders)
    paths = save_cohort_results(results_by_cohort, results_dir, "baseline")

    logger.info(f"Success! Batch evaluation complete. Results saved to {paths}")
    log_cohort_summaries(logger, results_by_cohort)


if __name__ == "__main__":
    main()
