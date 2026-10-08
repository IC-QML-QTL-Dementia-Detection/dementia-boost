"""Multiseed Quantum Transfer Learning (QTL) training with embedding caching.

This script selects the pre-trained classical CNN baseline with the best
validation metrics, extracts and caches the train and validation feature
representations once, and trains a hybrid Dressed Quantum Network (DQN)
classification head across multiple random seeds (0 to 100) using
BCEWithLogitsLoss on NIfTI axial slices. The test cohort is never loaded here.
Each run is described by a `RunSpec` (qubits, layers, ansatz, gradient method,
and the selected baseline as `backbone_id`); checkpoints, per-epoch histories,
and configuration are written where `ResultsLayout` puts them, and a run whose
checkpoint exists is skipped. Loss plots are rendered from the histories by
`scripts/viz/visualize_loss.py`.
"""

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.identity import Paradigm, RunSpec, label, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.embedding_cache import FeatureCacheManager
from dementia_boost.data.split_manifest import read_split_id
from dementia_boost.models.builder import (
    assemble_dementia_classifier,
    load_baseline_backbone,
)
from dementia_boost.models.quantum_cnn import QuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.selection import select_backbone
from dementia_boost.training.trainer import BaselineTrainer

DEFAULT_EXPERIMENT_SEEDS: range = range(0, 101)
DEFAULT_EPOCHS_PER_RUN: int = 100
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_LR_STEP_SIZE: int = 10
DEFAULT_LR_GAMMA: float = 0.75
DEFAULT_FEATURE_DIM: int = 2304
DEFAULT_ANSATZ: str = "paper"
DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 4
DEFAULT_TORCH_DEVICE: str = "cpu"
DEFAULT_QUANTUM_DEVICE: str = "lightning.qubit"
DEFAULT_GRADIENT_METHOD: str = "adjoint"
DEFAULT_EVAL_EVERY: int = 1


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


def build_spec(seed: int, backbone: RunSpec) -> RunSpec:
    """Builds the specification of one PennyLane QTL run on a selected baseline.

    Args:
        seed: Random seed of the run.
        backbone: Spec of the baseline run the head is built on.

    Returns:
        The QTL spec, trained on the same split as its backbone.
    """
    return RunSpec(
        paradigm=Paradigm.PL_QTL,
        ansatz=DEFAULT_ANSATZ,
        n_qubits=DEFAULT_N_QUBITS,
        n_layers=DEFAULT_N_LAYERS,
        lr=DEFAULT_LEARNING_RATE,
        lr_step_size=DEFAULT_LR_STEP_SIZE,
        lr_gamma=DEFAULT_LR_GAMMA,
        epochs=DEFAULT_EPOCHS_PER_RUN,
        batch_size=DEFAULT_BATCH_SIZE,
        gradient=DEFAULT_GRADIENT_METHOD,
        split_id=backbone.split_id,
        seed=seed,
        backbone_id=run_id(backbone),
    )


def main() -> None:
    """Executes Quantum Transfer Learning sweep across seeds on cached embeddings."""
    logger = setup_logger("qtl_multiseed_nifti")
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
    logger.info(f"Target PyTorch Device: {device}")
    logger.info(f"Target Quantum Device: {DEFAULT_QUANTUM_DEVICE}")
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

    logger.info(
        f"Beginning fast in-memory QTL sweep ({DEFAULT_N_QUBITS} qubits, "
        f"{DEFAULT_N_LAYERS} layers) across {len(DEFAULT_EXPERIMENT_SEEDS)} seeds..."
    )

    for seed in DEFAULT_EXPERIMENT_SEEDS:
        spec = build_spec(seed, backbone)

        if layout.is_done(spec):
            logger.info(
                f"Checkpoint already exists for {label(spec)} at "
                f"{layout.checkpoint_path(spec)}. Skipping execution."
            )
            continue

        logger.info(f"=== Starting QTL Experiment: {label(spec)} ===")

        set_seed(seed)

        head = QuantumClassifierHead(
            in_features=DEFAULT_FEATURE_DIM,
            n_qubits=DEFAULT_N_QUBITS,
            n_layers=DEFAULT_N_LAYERS,
            quantum_device=DEFAULT_QUANTUM_DEVICE,
            diff_method=DEFAULT_GRADIENT_METHOD,
        ).to(device)
        head.apply(QuantumClassifierHead.apply_glorot_init)

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
            extras={
                "torch_device": str(device),
                "quantum_device": DEFAULT_QUANTUM_DEVICE,
            },
        )

        trainer.train()
        logger.info(f"=== Completed QTL Experiment: {label(spec)} ===\n")

    logger.info("Quantum Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
