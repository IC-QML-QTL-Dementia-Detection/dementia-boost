"""Evaluation of saved checkpoints on the validation and test cohorts.

Every checkpoint is evaluated once per cohort. Validation metrics feed model
selection; test metrics are only reported. The two are written to separate
files so a consumer can never confuse them.
"""

import os
from collections.abc import Mapping, Sequence
from logging import Logger

from torch.utils.data import DataLoader

from dementia_boost.telemetry.metrics import EvaluationResult, MetricsAnalyzer
from dementia_boost.training.evaluator import ModelEvaluator


def cohort_results_path(results_dir: str, stem: str, cohort: str) -> str:
    """Builds the results file path for one cohort.

    Args:
        results_dir: Directory holding the metrics files.
        stem: Paradigm file stem, for example "baseline" or "tl".
        cohort: "val" or "test".

    Returns:
        `<results_dir>/<stem>_results.json` for the test cohort and
        `<results_dir>/<stem>_val_results.json` for the validation cohort.
    """
    suffix = "_results.json" if cohort == "test" else f"_{cohort}_results.json"
    return os.path.join(results_dir, f"{stem}{suffix}")


def evaluate_checkpoints(
    evaluator: ModelEvaluator,
    checkpoints: Sequence[tuple[str, str]],
    loaders: Mapping[str, DataLoader],
) -> dict[str, list[EvaluationResult]]:
    """Evaluates each checkpoint on every cohort loader.

    The weights of a checkpoint are loaded once and used for all cohorts.

    Args:
        evaluator: Evaluator holding the architecture the checkpoints fit.
        checkpoints: `(run_id, checkpoint path)` pairs, evaluated in order.
        loaders: Mapping of cohort name to its DataLoader.

    Returns:
        Mapping of cohort name to one `EvaluationResult` per checkpoint, in
        checkpoint order.
    """
    results: dict[str, list[EvaluationResult]] = {cohort: [] for cohort in loaders}
    for run_id, path in checkpoints:
        evaluator.load_weights(path)
        for cohort, loader in loaders.items():
            y_true, y_prob = evaluator.predict(loader)
            results[cohort].append(
                MetricsAnalyzer.calculate_metrics(run_id, y_true, y_prob)
            )
    return results


def log_cohort_summaries(
    logger: Logger, results_by_cohort: Mapping[str, list[EvaluationResult]]
) -> None:
    """Logs the mean and standard deviation of each metric, per cohort.

    Args:
        logger: Logger that receives one line per cohort and metric.
        results_by_cohort: Output of `evaluate_checkpoints`.
    """
    labels = (
        ("Acc", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1_score"),
        ("AUC", "auc"),
    )
    for cohort, results in results_by_cohort.items():
        stats = MetricsAnalyzer.aggregate_results(results)
        for label, key in labels:
            logger.info(
                f"[{cohort}] Mean {label}: {stats[key].mean:.4f} "
                f"\\pm {stats[key].std:.4f}"
            )


def save_cohort_results(
    results_by_cohort: Mapping[str, list[EvaluationResult]],
    results_dir: str,
    stem: str,
) -> dict[str, str]:
    """Aggregates and saves the results of each cohort to its own JSON file.

    Args:
        results_by_cohort: Output of `evaluate_checkpoints`.
        results_dir: Directory for the metrics files, created if missing.
        stem: Paradigm file stem used in the file names.

    Returns:
        Mapping of cohort name to the path written.
    """
    os.makedirs(results_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for cohort, results in results_by_cohort.items():
        path = cohort_results_path(results_dir, stem, cohort)
        MetricsAnalyzer.save_to_json(
            results, MetricsAnalyzer.aggregate_results(results), path, cohort=cohort
        )
        paths[cohort] = path
    return paths
