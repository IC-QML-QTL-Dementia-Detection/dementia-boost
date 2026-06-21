import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.builder import build_classical_tl_model
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.trainer import BaselineTrainer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logger = setup_logger("classical_tl_multiseed")
    device = get_device()

    baseline_dir = "./data/results/trained_models"
    tl_save_dir = "./data/results/trained_tl_models"
    epochs = 100
    batch_size = 64

    os.makedirs(tl_save_dir, exist_ok=True)

    baseline_files = [
        f
        for f in os.listdir(baseline_dir)
        if f.startswith("baseline_seed_") and f.endswith(".pt")
    ]
    if not baseline_files:
        logger.error(f"No baseline models found in {baseline_dir}")
        sys.exit(1)

    logger.info(f"Target Device: {device}")
    logger.info(
        f"Found {len(baseline_files)} baseline models."
        f"Beginning Transfer Learning sweep..."
    )

    loader_manager = OasisDataLoader(batch_size=batch_size)
    train_loader = loader_manager.get_data_loader(is_train=True)
    test_loader = loader_manager.get_data_loader(is_train=False)

    for file_name in baseline_files:
        seed_str = file_name.replace("baseline_seed_", "").replace(".pt", "")
        experiment_seed = int(seed_str)
        run_id = f"seed_{experiment_seed}"

        baseline_weights_path = os.path.join(baseline_dir, file_name)

        logger.info(f"=== Starting Classical Transfer Learning on {run_id} ===")

        set_seed(experiment_seed)
        model = build_classical_tl_model(baseline_weights_path, device)

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
            save_dir=tl_save_dir,
        )

        trainer.train(epochs=epochs, run_id=f"tl_{run_id}")
        logger.info(f"=== Completed Experiment: {run_id} ===\n")

    logger.info("Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
