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
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logger = setup_logger("qtl_single_run_jpg")
    device = get_device()

    baseline_dir = "./data/results/trained_models/jpg"
    qtl_save_dir = "./data/results/trained_qtl_models/jpg"

    epochs_per_run = 100
    batch_size = 64

    n_qubits = 6
    n_layers = 4

    os.makedirs(qtl_save_dir, exist_ok=True)

    experiment_seed = 1
    run_id = f"seed_{experiment_seed}"
    baseline_weights_path = os.path.join(baseline_dir, f"baseline_{run_id}.pt")

    if not os.path.exists(baseline_weights_path):
        logger.error(f"Baseline model for seed 1 not found at {baseline_weights_path}")
        sys.exit(1)

    logger.info(f"Target Device: {device}")
    logger.info(f"Beginning single QTL run for seed {experiment_seed}...")

    loader_manager = OasisDataLoader(batch_size=batch_size, mode="jpg")
    train_loader = loader_manager.get_data_loader(is_train=True)
    test_loader = loader_manager.get_data_loader(is_train=False)

    logger.info(
        f"=== Starting QTL on {run_id} ({n_qubits} qubits, {n_layers} layers) ==="
    )

    set_seed(experiment_seed)

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

    trainer.train(epochs=epochs_per_run, run_id=f"qtl_{run_id}")

    logger.info("Quantum Transfer Learning single run complete.")


if __name__ == "__main__":
    main()
