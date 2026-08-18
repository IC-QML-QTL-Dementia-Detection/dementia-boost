import json
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_best_baseline(metrics_json_path: str) -> str:
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(
            f"Baseline results JSON not found at: {metrics_json_path}"
        )

    with open(metrics_json_path) as f:
        data = json.load(f)

    individual_runs: list[dict[str, float | str]] = data.get("individual_runs", [])
    if not individual_runs:
        raise ValueError("No individual runs found in baseline telemetry JSON.")

    best_run = max(
        individual_runs,
        key=lambda r: (
            float(r.get("accuracy", 0.0)),
            float(r.get("f1_score", 0.0)),
            float(r.get("auc", 0.0)),
        ),
    )

    return str(best_run["run_id"])


def main() -> None:
    logger = setup_logger("ctl_multiseed_jpg")
    device = get_device()

    baseline_metrics_path = "./data/results/metrics/jpg/baseline_results.json"
    baseline_dir = "./data/results/trained_models/jpg"
    ctl_save_dir = "./data/results/trained_tl_models/jpg"

    experiment_seeds = range(1, 101)
    epochs_per_run = 100
    batch_size = 64

    os.makedirs(ctl_save_dir, exist_ok=True)

    try:
        best_run_id = select_best_baseline(baseline_metrics_path)
    except Exception as e:
        logger.error(f"Failed to select optimal baseline: {e}")
        sys.exit(1)

    baseline_weights_path = os.path.join(baseline_dir, f"{best_run_id}.pt")
    if not os.path.exists(baseline_weights_path):
        logger.error(
            f"Selected optimal baseline weights missing at: {baseline_weights_path}"
        )
        sys.exit(1)

    logger.info(f"Target Device: {device}")
    logger.info(
        f"Selected Optimal Baseline Backbone: '{best_run_id}' ({baseline_weights_path})"
    )
    logger.info(f"Beginning CTL sweep across {len(experiment_seeds)} seeds...")

    loader_manager = OasisDataLoader(batch_size=batch_size, mode="jpg")
    train_loader = loader_manager.get_data_loader(is_train=True)
    test_loader = loader_manager.get_data_loader(is_train=False)

    for seed in experiment_seeds:
        run_id = f"tl_seed_{seed}"
        logger.info(
            f"=== Starting CTL Experiment: {run_id} (Backbone: {best_run_id}) ==="
        )

        set_seed(seed)

        model = build_classical_tl_model(
            baseline_weights_path=baseline_weights_path,
            device=device,
            use_sigmoid=False,
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
            save_dir=ctl_save_dir,
        )

        trainer.train(epochs=epochs_per_run, run_id=run_id)
        logger.info(f"=== Completed CTL Experiment: {run_id} ===\n")

    logger.info("Classical Transfer Learning multi-seed sweep complete.")


if __name__ == "__main__":
    main()
