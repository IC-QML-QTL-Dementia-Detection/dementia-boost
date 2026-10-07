"""Batch evaluation engine for Classical Transfer Learning models on NIfTI data.

This script iterates across all saved Classical Transfer Learning (CTL)
model checkpoints (.pt) trained on NIfTI axial slices, performs batched inference
on the validation and test cohorts, calculates full classification metrics
(Accuracy, Precision, Recall, F1-score, AUC-ROC, log loss, Confusion Matrix),
computes statistical distributions, and exports one telemetry JSON per cohort.
Validation metrics are only for model selection; test metrics are only reported.
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
    """Executes the batch evaluation pipeline for CTL NIfTI models."""
    logger = setup_logger("evaluate_tl_nifti")
    device = get_device()

    models_dir = "./data/results/trained_tl_models/nifti"
    results_dir = "./data/results/metrics/nifti"

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(models_dir):
        logger.error(f"Checkpoint directory {models_dir} not found.")
        sys.exit(1)

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("baseline_tl_") and f.endswith(".pt")
    ]
    if not model_files:
        logger.error(f"No CTL model checkpoint files found in {models_dir}.")
        sys.exit(1)

    model_files.sort(
        key=lambda f: (
            int(f.replace("baseline_tl_seed_", "").replace(".pt", ""))
            if f.replace("baseline_tl_seed_", "").replace(".pt", "").isdigit()
            else f
        )
    )

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

    logger.info(f"Found {len(model_files)} CTL models. Beginning evaluation...")

    checkpoints = [
        (file_name.replace(".pt", ""), os.path.join(models_dir, file_name))
        for file_name in model_files
    ]
    results_by_cohort = evaluate_checkpoints(evaluator, checkpoints, loaders)
    paths = save_cohort_results(results_by_cohort, results_dir, "tl")

    logger.info(f"Success! CTL evaluation complete. Results saved to {paths}")
    log_cohort_summaries(logger, results_by_cohort)


if __name__ == "__main__":
    main()
