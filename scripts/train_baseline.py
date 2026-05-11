import sys

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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    set_seed(42)
    logger = setup_logger("baseline_train")

    logger.info("Initializing Baseline Training Prototype")

    device = get_device()
    logger.info(f"Target Device: {device}")

    loader_manager = OasisDataLoader(batch_size=4)
    try:
        train_loader = loader_manager.get_data_loader(is_train=True)
        test_loader = loader_manager.get_data_loader(is_train=False)
        logger.info(
            f"Data loaded: {len(train_loader)} training batches, "
            f"{len(test_loader)} test batches."
        )
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    extractor = LeNetFeatureExtractor()
    head = ClassicalClassifierHead()
    model = DementiaClassifier(feature_extractor=extractor, classifier_head=head)

    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.75)

    epochs = 5

    print()

    logger.info("Starting Training")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.round()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}"
        )

    print()
    logger.info("Training Complete")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predictions = outputs.round()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    logger.info(f"Final Test Loss: {val_loss / val_total:.4f}")
    logger.info(f"Final Test Acc: {val_correct / val_total:.4f}")


if __name__ == "__main__":
    main()
