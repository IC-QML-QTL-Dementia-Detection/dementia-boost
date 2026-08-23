"""Multiseed Quantum Transfer Learning (QTL) training on optimal NIfTI baseline.

This script identifies the optimal pre-trained classical CNN baseline from
telemetry metrics, freezes its convolutional feature extractor, and trains a
hybrid Dressed Quantum Network (DQN) classification head across multiple random
seeds (0 to 100) using BCEWithLogitsLoss on NIfTI axial slices.
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
from dementia_boost.models.builder import build_quantum_tl_model
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.trainer import BaselineTrainer


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


def select_best_baseline(metrics_json_path: str) -> str:
    """Identifies the optimal baseline model run from telemetry results JSON.

    Selects the run maximizing Accuracy, with F1-score and AUC as tie-breakers.

    Args:
        metrics_json_path: Filepath to the serialized baseline results JSON.

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
    """Executes Quantum Transfer Learning sweep across seeds on the optimal baseline."""
    logger = setup_logger("qtl_multiseed_nifti")
    device = get_device()

    baseline_metrics_path = "./data/results/metrics/nifti/baseline_results.json"
    baseline_dir = "./data/results/trained_models/nifti"
    qtl_save_dir = "./data/results/trained_qtl_models/nifti"

    experiment_seeds = range(0, 101)
    epochs_per_run = 100
    batch_size = 64

    n_qubits = 6
    n_layers = 4

    os.makedirs(qtl_save_dir, exist_ok=True)

    try:
        best_run_id = select_best_baseline(baseline_metrics_path)
    except Exception as error:
        logger.error(f"Failed to select optimal baseline model: {error}")
        sys.exit(1)

    checkpoint_name = (
        best_run_id if best_run_id.endswith(".pt") else f"{best_run_id}.pt"
    )
    baseline_weights_path = os.path.join(baseline_dir, checkpoint_name)

    if not os.path.exists(baseline_weights_path):
        logger.error(
            f"Selected optimal baseline weights missing at: {baseline_weights_path}"
        )
        sys.exit(1)

    logger.info(f"Target Device: {device}")
    logger.info(
        f"Selected Optimal Baseline Backbone: '{best_run_id}' ({baseline_weights_path})"
    )
    logger.info(
        f"Beginning QTL sweep ({n_qubits} qubits, {n_layers} layers) "
        f"across {len(experiment_seeds)} seeds..."
    )

    loader_manager = OasisDataLoader(batch_size=batch_size, mode="nifti")
    train_loader = loader_manager.get_data_loader(is_train=True)
    test_loader = loader_manager.get_data_loader(is_train=False)

    for seed in experiment_seeds:
        run_id = f"qtl_seed_{seed}"
        logger.info(
            f"=== Starting QTL Experiment: {run_id} (Backbone: {best_run_id}) ==="
        )

        set_seed(seed)

        model = build_quantum_tl_model(
            baseline_weights_path=baseline_weights_path,
            device=device,
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=1e-4)

        criterion = nn.BCEWithLogitsLoss()
        scheduler = StepLR(optimizer, step_size=10, gamma=0.75)

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=qtl_save_dir,
        )

        trainer.train(epochs=epochs_per_run, run_id=run_id)
        logger.info(f"=== Completed QTL Experiment: {run_id} ===\n")

    logger.info("Quantum Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
