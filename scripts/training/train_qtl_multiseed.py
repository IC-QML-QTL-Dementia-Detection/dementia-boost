"""Multiseed Quantum Transfer Learning (QTL) training with embedding caching.

This script identifies the optimal pre-trained classical CNN baseline from
telemetry metrics, extracts and caches feature representations once, and
trains a hybrid Dressed Quantum Network (DQN) classification head across
multiple random seeds (0 to 100) using BCEWithLogitsLoss on NIfTI axial slices.
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.embedding_cache import FeatureCacheManager
from dementia_boost.models.builder import (
    assemble_dementia_classifier,
    load_baseline_backbone,
)
from dementia_boost.models.quantum_cnn import QuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.trainer import BaselineTrainer

DEFAULT_BASELINE_METRICS_PATH: str = (
    "./data/results/metrics/nifti/baseline_results.json"
)
DEFAULT_BASELINE_DIR: str = "./data/results/trained_models/nifti"
DEFAULT_QTL_SAVE_DIR: str = "./data/results/trained_qtl_models/nifti"
DEFAULT_EXPERIMENT_SEEDS: range = range(0, 101)
DEFAULT_EPOCHS_PER_RUN: int = 100
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_LR_STEP_SIZE: int = 10
DEFAULT_LR_GAMMA: float = 0.75
DEFAULT_FEATURE_DIM: int = 2304
DEFAULT_N_QUBITS: int = 6
DEFAULT_N_LAYERS: int = 4
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


def select_best_baseline(metrics_json_path: str = DEFAULT_BASELINE_METRICS_PATH) -> str:
    """Identifies the optimal baseline model run from telemetry results JSON.

    Selects the run maximizing Accuracy, with F1-score and AUC as tie-breakers.

    Args:
        metrics_json_path: Filepath to the serialized baseline results JSON.
            Defaults to DEFAULT_BASELINE_METRICS_PATH.

    Returns:
        The run_id string of the top-performing baseline model.

    Raises:
        FileNotFoundError: If the metrics JSON file does not exist.
        ValueError: If no individual runs are present in the JSON payload.
    """
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(
            f"Baseline results JSON not found at: {metrics_json_path}"
        )

    with open(metrics_json_path) as file:
        data = json.load(file)

    individual_runs: list[dict[str, float | str]] = data.get("individual_runs", [])
    if not individual_runs:
        raise ValueError("No individual runs found in baseline telemetry JSON.")

    best_run = max(
        individual_runs,
        key=lambda item: (
            float(item.get("accuracy", 0.0)),
            float(item.get("f1_score", 0.0)),
            float(item.get("auc", 0.0)),
        ),
    )

    return str(best_run["run_id"])


def main() -> None:
    """Executes Quantum Transfer Learning sweep across seeds on cached embeddings."""
    logger = setup_logger("qtl_multiseed_nifti")
    device = get_device()

    os.makedirs(DEFAULT_QTL_SAVE_DIR, exist_ok=True)

    try:
        best_run_id = select_best_baseline(DEFAULT_BASELINE_METRICS_PATH)
    except Exception as error:
        logger.error(f"Failed to select optimal baseline model: {error}")
        sys.exit(1)

    checkpoint_name = (
        best_run_id if best_run_id.endswith(".pt") else f"{best_run_id}.pt"
    )
    baseline_weights_path = os.path.join(DEFAULT_BASELINE_DIR, checkpoint_name)

    if not os.path.exists(baseline_weights_path):
        logger.error(
            f"Selected optimal baseline weights missing at: {baseline_weights_path}"
        )
        sys.exit(1)

    logger.info(f"Target PyTorch Device: {device}")
    logger.info(f"Target Quantum Device: {DEFAULT_QUANTUM_DEVICE}")
    logger.info(
        f"Selected Optimal Baseline Backbone: '{best_run_id}' ({baseline_weights_path})"
    )
    logger.info(
        f"Beginning QTL sweep ({DEFAULT_N_QUBITS} qubits, {DEFAULT_N_LAYERS} layers) "
        f"across {len(DEFAULT_EXPERIMENT_SEEDS)} seeds..."
    )

    feature_extractor = load_baseline_backbone(baseline_weights_path, device)

    raw_loader_manager = OasisDataLoader(batch_size=DEFAULT_BATCH_SIZE, mode="nifti")
    raw_train_loader = raw_loader_manager.get_data_loader(is_train=True)
    raw_test_loader = raw_loader_manager.get_data_loader(is_train=False)

    logger.info("Extracting and caching training embeddings from baseline backbone...")
    train_features, train_labels = FeatureCacheManager.extract_features(
        feature_extractor=feature_extractor,
        data_loader=raw_train_loader,
        device=device,
    )
    logger.info("Extracting and caching test embeddings from baseline backbone...")
    test_features, test_labels = FeatureCacheManager.extract_features(
        feature_extractor=feature_extractor,
        data_loader=raw_test_loader,
        device=device,
    )

    train_loader = FeatureCacheManager.create_cached_loader(
        features=train_features,
        labels=train_labels,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = FeatureCacheManager.create_cached_loader(
        features=test_features,
        labels=test_labels,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
    )

    logger.info(
        "Beginning fast in-memory QTL sweep across "
        f"{len(DEFAULT_EXPERIMENT_SEEDS)} seeds..."
    )

    for seed in DEFAULT_EXPERIMENT_SEEDS:
        run_id = f"qtl_seed_{seed}"
        checkpoint_path = os.path.join(DEFAULT_QTL_SAVE_DIR, f"baseline_{run_id}.pt")

        if os.path.exists(checkpoint_path):
            logger.info(
                f"Checkpoint already exists for {run_id} at {checkpoint_path}. "
                "Skipping execution."
            )
            continue

        logger.info(
            f"=== Starting QTL Experiment: {run_id} (Backbone: {best_run_id}) ==="
        )

        set_seed(seed)

        head = QuantumClassifierHead(
            in_features=DEFAULT_FEATURE_DIM,
            n_qubits=DEFAULT_N_QUBITS,
            n_layers=DEFAULT_N_LAYERS,
            quantum_device=DEFAULT_QUANTUM_DEVICE,
        ).to(device)
        head.apply(QuantumClassifierHead.apply_glorot_init)

        full_model = assemble_dementia_classifier(
            feature_extractor=feature_extractor,
            classifier_head=head,
        )

        optimizer = optim.Adam(head.parameters(), lr=DEFAULT_LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = StepLR(
            optimizer,
            step_size=DEFAULT_LR_STEP_SIZE,
            gamma=DEFAULT_LR_GAMMA,
        )

        trainer = BaselineTrainer(
            model=head,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=DEFAULT_QTL_SAVE_DIR,
            save_model=full_model,
        )

        trainer.train(epochs=DEFAULT_EPOCHS_PER_RUN, run_id=run_id)
        logger.info(f"=== Completed QTL Experiment: {run_id} ===\n")

    logger.info("Quantum Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
