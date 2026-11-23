"""Multiseed Classical Transfer Learning (CTL) training with embedding caching.

This script reads the baseline validation results, selects the baseline run with
the best validation metrics, extracts and caches the train and validation feature
representations once, and trains newly initialized classical dense heads across
multiple random seeds (0 to 100) using BCEWithLogitsLoss. The test cohort is
never loaded here. Each run is described by a `RunSpec` that records the
selected baseline as its `backbone_id`; checkpoints, per-epoch histories, and
configuration are written where `ResultsLayout` puts them, and a run whose
checkpoint exists is skipped. Loss plots are rendered from the histories by
`scripts/viz/visualize_loss.py`.

`--head` picks the classical head from the registry (the base paper's head by
default, or `tanh_bottleneck`, the quantum head's classical layers without the
circuit), `--hidden_features` sets its hidden width, and `--seeds N` runs only
the first N seeds.
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.identity import (
    DEFAULT_CLASSICAL_HEAD,
    Paradigm,
    RunSpec,
    label,
    run_id,
)
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.embedding_cache import FeatureCacheManager
from dementia_boost.data.split_manifest import read_split_id
from dementia_boost.models.builder import (
    assemble_dementia_classifier,
    load_baseline_backbone,
)
from dementia_boost.models.classical_cnn import (
    CLASSICAL_HEADS,
    ClassicalClassifierHead,
    classical_head_for,
    stored_hidden_features,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.selection import select_backbone
from dementia_boost.training.trainer import BaselineTrainer

DEFAULT_EXPERIMENT_SEEDS: range = range(0, 101)
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


def _positive_int(text: str) -> int:
    """Parses a strictly positive integer for the command line.

    Args:
        text: The argument as typed.

    Returns:
        The integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}")
    return value


def parse_arguments() -> argparse.Namespace:
    """Parses the command-line options of the sweep.

    Returns:
        The namespace with `head` (the classical head name), `hidden_features`
        (its width, or None for the head's default), and `seeds` (how many of
        the default seeds to run, or None for all).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head",
        choices=sorted(CLASSICAL_HEADS),
        default=DEFAULT_CLASSICAL_HEAD,
        help="Classical head on the frozen backbone; part of the configuration.",
    )
    parser.add_argument(
        "--hidden_features",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Hidden width of the head (default: the head's own, 5 for paper and "
        "6 for tanh_bottleneck); part of the configuration.",
    )
    parser.add_argument(
        "--seeds",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Run only the first N seeds. Extending later reuses the same "
        "configuration.",
    )
    return parser.parse_args()


def build_spec(
    seed: int,
    backbone: RunSpec,
    head: str = DEFAULT_CLASSICAL_HEAD,
    hidden_features: int | None = None,
) -> RunSpec:
    """Builds the specification of one CTL run on a selected baseline.

    Args:
        seed: Random seed of the run.
        backbone: Spec of the baseline run the head is built on.
        head: Registry name of the classical head.
        hidden_features: Hidden width of the head, or None for its default.

    Returns:
        The CTL spec, trained on the same split as its backbone.
    """
    return RunSpec(
        paradigm=Paradigm.CTL,
        lr=DEFAULT_LEARNING_RATE,
        lr_step_size=DEFAULT_LR_STEP_SIZE,
        lr_gamma=DEFAULT_LR_GAMMA,
        epochs=DEFAULT_EPOCHS_PER_RUN,
        batch_size=DEFAULT_BATCH_SIZE,
        split_id=backbone.split_id,
        seed=seed,
        backbone_id=run_id(backbone),
        head=head,
        hidden_features=stored_hidden_features(head, hidden_features),
    )


def main() -> None:
    """Executes Classical Transfer Learning on cached baseline embeddings."""
    arguments = parse_arguments()
    seeds = DEFAULT_EXPERIMENT_SEEDS[: arguments.seeds]
    logger = setup_logger("classical_ctl_multiseed_nifti")
    device = get_device()
    layout = ResultsLayout()

    try:
        backbone = select_backbone(layout)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to select the baseline backbone: {error}")
        sys.exit(1)

    split_id = read_split_id(OasisDataLoader.RESULTS_PATH)
    if backbone.split_id != split_id:
        logger.error(
            f"The selected baseline was trained on split {backbone.split_id}, but "
            f"the data on disk is split {split_id}. Retrain the baselines."
        )
        sys.exit(1)

    baseline_weights_path = layout.checkpoint_path(backbone)
    logger.info(f"Target Device: {device}")
    logger.info(
        f"Selected Baseline Backbone: {label(backbone)} ({baseline_weights_path})"
    )

    feature_extractor = load_baseline_backbone(baseline_weights_path, device)

    raw_loader_manager = OasisDataLoader(batch_size=DEFAULT_BATCH_SIZE)
    raw_train_loader = raw_loader_manager.get_data_loader("train")
    raw_val_loader = raw_loader_manager.get_data_loader("val")

    logger.info("Extracting and caching training embeddings from baseline backbone...")
    train_features, train_labels = FeatureCacheManager.extract_features(
        feature_extractor=feature_extractor,
        data_loader=raw_train_loader,
        device=device,
    )
    logger.info(
        "Extracting and caching validation embeddings from baseline backbone..."
    )
    val_features, val_labels = FeatureCacheManager.extract_features(
        feature_extractor=feature_extractor,
        data_loader=raw_val_loader,
        device=device,
    )

    train_loader = FeatureCacheManager.create_cached_loader(
        features=train_features,
        labels=train_labels,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = FeatureCacheManager.create_cached_loader(
        features=val_features,
        labels=val_labels,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
    )

    logger.info(f"Beginning fast in-memory CTL sweep across {len(seeds)} seeds...")

    for seed in seeds:
        spec = build_spec(seed, backbone, arguments.head, arguments.hidden_features)

        if layout.is_done(spec):
            logger.info(
                f"Checkpoint already exists for {label(spec)} at "
                f"{layout.checkpoint_path(spec)}. Skipping execution."
            )
            continue

        logger.info(f"=== Starting CTL Experiment: {label(spec)} ===")

        set_seed(seed)

        head = classical_head_for(spec).to(device)
        head.apply(ClassicalClassifierHead.apply_glorot_init)

        full_model = assemble_dementia_classifier(
            feature_extractor=feature_extractor,
            classifier_head=head,
        )

        optimizer = optim.Adam(head.parameters(), lr=spec.lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = StepLR(optimizer, step_size=spec.lr_step_size, gamma=spec.lr_gamma)

        trainer = BaselineTrainer(
            model=head,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            spec=spec,
            layout=layout,
            save_model=full_model,
            eval_every=DEFAULT_EVAL_EVERY,
            extras={"torch_device": str(device)},
        )

        trainer.train()
        logger.info(f"=== Completed CTL Experiment: {label(spec)} ===\n")

    logger.info("Classical Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
