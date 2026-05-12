import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dementia_boost.core.reproducibility import set_seed
from dementia_boost.data import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.training.trainer import BaselineTrainer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logger = setup_logger("baseline_train")
    device = get_device()

    logger.info(f"Target Device: {device}")

    loader_manager = OasisDataLoader(batch_size=4)
    train_loader = loader_manager.get_data_loader(is_train=True)
    test_loader = loader_manager.get_data_loader(is_train=False)
    logger.info(
        f"Data loaded: {len(train_loader)} training batches, "
        f"{len(test_loader)} test batches."
    )

    logger.info("Starting Baseline Training")
    set_seed(42)

    extractor = LeNetFeatureExtractor()
    head = ClassicalClassifierHead()
    model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.75)

    epochs = 5

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        save_dir="./data/results/trained_models",
    )

    trainer.train(epochs=epochs, run_id="42")
    logger.info("Completed Experiment")


if __name__ == "__main__":
    main()
