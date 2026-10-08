"""Batch evaluation engine for Classical Transfer Learning models on NIfTI data.

This script finds every finished Classical Transfer Learning (CTL) run through
its training history, performs batched inference on the validation and test
cohorts for each configuration, calculates full classification metrics
(Accuracy, Precision, Recall, F1-score, AUC-ROC, log loss, Confusion Matrix),
computes statistical distributions, and exports one telemetry JSON per cohort.
Validation metrics are only for model selection; test metrics are only reported.
"""

import sys

import torch

from dementia_boost.core.identity import Paradigm, RunSpec
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.checkpoint_evaluation import evaluate_paradigm

DEFAULT_BATCH_SIZE: int = 64


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


def build_model(spec: RunSpec) -> DementiaClassifier:
    """Builds the CTL architecture the checkpoints of a configuration fit.

    Args:
        spec: Spec of one run of the configuration (unused: the classical head
            has no configurable size).

    Returns:
        An untrained classifier with a classical head.
    """
    return DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )


def main() -> None:
    """Executes the batch evaluation pipeline for CTL NIfTI models."""
    logger = setup_logger("evaluate_ctl_nifti")
    device = get_device()

    loader_manager = OasisDataLoader(batch_size=DEFAULT_BATCH_SIZE)
    loaders = {
        cohort: loader_manager.get_data_loader(cohort) for cohort in ("val", "test")
    }

    outcome = evaluate_paradigm(
        ResultsLayout(), Paradigm.CTL, build_model, loaders, device, logger
    )
    if not outcome:
        logger.error("No finished CTL runs found. Run train_ctl_multiseed.py first.")
        sys.exit(1)

    logger.info(
        f"Success! CTL evaluation complete for {len(outcome)} configuration(s)."
    )


if __name__ == "__main__":
    main()
