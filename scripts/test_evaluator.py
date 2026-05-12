import os
import sys

import torch

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.metrics import MetricsAnalyzer
from dementia_boost.training.evaluator import ModelEvaluator


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logger = setup_logger("test_evaluator")
    device = get_device()
    models_dir = "./data/results/trained_models"

    if not os.path.exists(models_dir):
        logger.error(f"Directory {models_dir} not found.")
        sys.exit(1)

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not model_files:
        logger.error("No .pt files found in trained_models/")
        sys.exit(1)

    target_file = os.path.join(models_dir, model_files[0])
    logger.info(f"Testing Evaluator on: {target_file}")

    base_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(),
    )

    loader_manager = OasisDataLoader(batch_size=64)
    test_loader = loader_manager.get_data_loader(is_train=False)

    evaluator = ModelEvaluator(model=base_model, device=device)
    evaluator.load_weights(target_file)

    logger.info("Running inference over test set...")
    y_true, y_prob = evaluator.predict(test_loader)

    logger.info("Calculating metrics...")
    metrics = MetricsAnalyzer.calculate_metrics(y_true, y_prob)

    logger.info("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        logger.info(f"{key.capitalize():<16}: {value}")


if __name__ == "__main__":
    main()
