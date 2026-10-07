"""Evaluation of saved checkpoints on the validation and test cohorts.

Every checkpoint is evaluated once per cohort. Validation metrics feed model
selection; test metrics are only reported. The two are written to separate files
per configuration, each carrying the configuration it was computed for, so a
consumer can never confuse cohorts or mix configurations.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dementia_boost.core.identity import Paradigm, RunSpec, run_id
from dementia_boost.core.layout import ResultsLayout, config_payload
from dementia_boost.telemetry.metrics import EvaluationResult, MetricsAnalyzer
from dementia_boost.telemetry.run_listing import group_by_config, load_runs
from dementia_boost.training.evaluator import ModelEvaluator


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
    for checkpoint_run_id, path in checkpoints:
        evaluator.load_weights(path)
        for cohort, loader in loaders.items():
            y_true, y_prob = evaluator.predict(loader)
            results[cohort].append(
                MetricsAnalyzer.calculate_metrics(checkpoint_run_id, y_true, y_prob)
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


def save_configuration_results(
    layout: ResultsLayout,
    spec: RunSpec,
    results_by_cohort: Mapping[str, list[EvaluationResult]],
) -> dict[str, str]:
    """Aggregates and saves one configuration's results, a file per cohort.

    Args:
        layout: The results layout that decides the file paths.
        spec: The spec of any run of the configuration.
        results_by_cohort: Output of `evaluate_checkpoints` for that
            configuration's runs.

    Returns:
        Mapping of cohort name to the path written.
    """
    configuration = config_payload(spec)
    paths: dict[str, str] = {}
    for cohort, results in results_by_cohort.items():
        path = layout.metrics_path(spec.paradigm, configuration["config_id"], cohort)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        MetricsAnalyzer.save_to_json(
            results,
            MetricsAnalyzer.aggregate_results(results),
            path,
            cohort=cohort,
            configuration=configuration,
        )
        paths[cohort] = path
    return paths


def evaluate_paradigm(
    layout: ResultsLayout,
    paradigm: Paradigm | str,
    build_model: Callable[[RunSpec], nn.Module],
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    logger: Logger,
    configuration: str | None = None,
) -> dict[str, dict[str, list[EvaluationResult]]]:
    """Evaluates every finished run of a paradigm, configuration by configuration.

    Runs are found through their histories. A run whose checkpoint is missing
    (it did not finish) is left out. The model of each configuration is built
    from its own spec, so its architecture never comes from a constant.

    Args:
        layout: The results layout to read runs from and write results to.
        paradigm: The paradigm to evaluate.
        build_model: Builds the model architecture a configuration's checkpoints
            fit, from the spec of one of its runs.
        loaders: Mapping of cohort name to its DataLoader.
        device: Device to run inference on.
        logger: Logger for progress and summaries.
        configuration: Restrict to one `config_id`. Defaults to all.

    Returns:
        Mapping of `config_id` to the per-cohort results written for it.
        Empty if there is nothing to evaluate.
    """
    outcome: dict[str, dict[str, list[EvaluationResult]]] = {}
    groups = group_by_config(load_runs(layout, paradigm, configuration))

    for config, histories in groups.items():
        specs = [history.spec for history in histories]
        finished = [spec for spec in specs if layout.is_done(spec)]
        if len(finished) < len(specs):
            logger.warning(
                f"Configuration {config}: {len(specs) - len(finished)} run(s) "
                "without a checkpoint are left out."
            )
        if not finished:
            continue

        logger.info(f"Evaluating {len(finished)} run(s) of configuration {config}...")
        evaluator = ModelEvaluator(model=build_model(finished[0]), device=device)
        checkpoints = [(run_id(s), layout.checkpoint_path(s)) for s in finished]
        results = evaluate_checkpoints(evaluator, checkpoints, loaders)
        paths = save_configuration_results(layout, finished[0], results)

        logger.info(f"Results saved to {paths}")
        log_cohort_summaries(logger, results)
        outcome[config] = results

    return outcome
