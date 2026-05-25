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
    logger = setup_logger("evaluate_baseline")
    device = get_device()
    models_dir = "./data/results/trained_models"
    results_dir = "./data/results/metrics"

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(models_dir):
        logger.error(f"Directory {models_dir} not found.")
        sys.exit(1)

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not model_files:
        logger.error("No .pt files found.")
        sys.exit(1)

    base_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(),
    )

    batch_size = 64
    loader_manager = OasisDataLoader(batch_size=batch_size)
    test_loader = loader_manager.get_data_loader(is_train=False)
    evaluator = ModelEvaluator(model=base_model, device=device)

    all_results = []
    logger.info(f"Found {len(model_files)} models. Beginning batch evaluation...")

    for file_name in model_files:
        run_id = file_name.replace(".pt", "")
        file_path = os.path.join(models_dir, file_name)

        logger.info(f"Evaluating: {run_id}...")
        evaluator.load_weights(file_path)

        y_true, y_prob = evaluator.predict(test_loader)

        result_dto = MetricsAnalyzer.calculate_metrics(run_id, y_true, y_prob)
        all_results.append(result_dto)

    logger.info("Aggregating statistics across all runs...")
    aggregated_stats = MetricsAnalyzer.aggregate_results(all_results)

    output_json = os.path.join(results_dir, "baseline_results.json")
    MetricsAnalyzer.save_to_json(all_results, aggregated_stats, output_json)

    logger.info(f"Success! Batch evaluation complete. Results saved to {output_json}")

    logger.info(
        f"Mean Acc: {aggregated_stats['accuracy'].mean:.4f} "
        f"\\pm {aggregated_stats['accuracy'].std:.4f}"
    )
    logger.info(
        f"Mean Precision: {aggregated_stats['precision'].mean:.4f} "
        f"\\pm {aggregated_stats['precision'].std:.4f}"
    )
    logger.info(
        f"Mean Recall: {aggregated_stats['recall'].mean:.4f} "
        f"\\pm {aggregated_stats['recall'].std:.4f}"
    )
    logger.info(
        f"Mean F1: {aggregated_stats['f1_score'].mean:.4f} "
        f"\\pm {aggregated_stats['f1_score'].std:.4f}"
    )
    logger.info(
        f"Mean AUC: {aggregated_stats['auc'].mean:.4f} "
        f"\\pm {aggregated_stats['auc'].std:.4f}"
    )


if __name__ == "__main__":
    main()
