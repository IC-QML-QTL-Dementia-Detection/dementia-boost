"""Multiseed training script for classical baseline CNN on NIfTI MRI data.

This script trains the baseline LeNet-based Convolutional Neural Network across
100 random seeds (1 to 100) using raw logits output and BCEWithLogitsLoss on
preprocessed NIfTI axial slices. Training is monitored on the validation cohort
only; the test cohort is never loaded here. Each run is described by a `RunSpec`
(which records the split it trained on); its checkpoint, per-epoch history, and
configuration are written where `ResultsLayout` puts them, and a run whose
checkpoint exists is skipped. Loss plots are rendered from the histories by
`scripts/viz/visualize_loss.py`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.identity import Paradigm, RunSpec, label
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data import OasisDataLoader
from dementia_boost.data.split_manifest import read_split_id
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.trainer import BaselineTrainer

DEFAULT_EXPERIMENT_SEEDS: range = range(1, 101)
DEFAULT_EPOCHS_PER_RUN: int = 100
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_LR_STEP_SIZE: int = 10
DEFAULT_LR_GAMMA: float = 0.75
DEFAULT_EVAL_EVERY: int = 1


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


def build_spec(seed: int, split_id: str) -> RunSpec:
    """Builds the specification of one baseline run.

    Args:
        seed: Random seed of the run.
        split_id: ID of the data split the run trains on.

    Returns:
        The baseline spec.
    """
    return RunSpec(
        paradigm=Paradigm.BASELINE,
        lr=DEFAULT_LEARNING_RATE,
        lr_step_size=DEFAULT_LR_STEP_SIZE,
        lr_gamma=DEFAULT_LR_GAMMA,
        epochs=DEFAULT_EPOCHS_PER_RUN,
        batch_size=DEFAULT_BATCH_SIZE,
        split_id=split_id,
        seed=seed,
    )


def main() -> None:
    """Executes the multiseed training loop for the classical CNN on NIfTI data."""
    logger = setup_logger("baseline_train_nifti")
    device = get_device()
    logger.info(f"Target Device: {device}")

    loader_manager = OasisDataLoader(batch_size=DEFAULT_BATCH_SIZE)
    train_loader = loader_manager.get_data_loader("train")
    val_loader = loader_manager.get_data_loader("val")
    logger.info(
        f"Data loaded: {len(train_loader)} training batches, "
        f"{len(val_loader)} validation batches."
    )

    layout = ResultsLayout()
    split_id = read_split_id(OasisDataLoader.RESULTS_PATH)
    logger.info(f"Split: {split_id}")

    for seed in DEFAULT_EXPERIMENT_SEEDS:
        spec = build_spec(seed, split_id)

        if layout.is_done(spec):
            logger.info(
                f"Checkpoint already exists for {label(spec)} at "
                f"{layout.checkpoint_path(spec)}. Skipping execution."
            )
            continue

        logger.info(f"=== Starting Experiment: {label(spec)} ===")
        set_seed(seed)

        model = DementiaClassifier(
            feature_extractor=LeNetFeatureExtractor(),
            classifier_head=ClassicalClassifierHead(use_sigmoid=False),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=spec.lr)
        scheduler = StepLR(optimizer, step_size=spec.lr_step_size, gamma=spec.lr_gamma)

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            spec=spec,
            layout=layout,
            eval_every=DEFAULT_EVAL_EVERY,
            extras={"torch_device": str(device)},
        )

        trainer.train()
        logger.info(f"=== Completed Experiment: {label(spec)} ===\n")


if __name__ == "__main__":
    main()
