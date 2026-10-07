"""Model selection on validation metrics only.

Choosing a model with the test cohort, and then reporting that same cohort,
makes the reported numbers optimistic. Everything here reads validation
metrics: `select_best_baseline` refuses any results file that was not computed
on the validation cohort.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from dementia_boost.core.identity import Paradigm, RunSpec
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.run_listing import load_runs

RANKING_METRICS = ("auc", "f1_score", "log_loss")
VALIDATION_COHORT = "val"


def select_best_run(runs: Sequence[Mapping[str, Any]]) -> str:
    """Picks the best run by validation metrics.

    Runs are ranked by AUC-ROC (threshold-free, and the least sensitive to class
    balance on a small cohort), then F1-score, then lowest log loss. A complete
    tie is resolved by the smallest run ID, so the result never depends on the
    order of `runs`. Only the three ranking metrics are read.

    Args:
        runs: Per-run validation metrics, each with `run_id`, `auc`, `f1_score`,
            and `log_loss`.

    Returns:
        The run ID of the best run.

    Raises:
        ValueError: If `runs` is empty or a run lacks a ranking metric.
    """
    if not runs:
        raise ValueError("No runs to select from.")

    for run in runs:
        missing = [name for name in RANKING_METRICS if name not in run]
        if missing:
            raise ValueError(
                f"Run {run.get('run_id')!r} lacks validation metrics: {missing}"
            )

    best = min(
        runs,
        key=lambda run: (
            -float(run["auc"]),
            -float(run["f1_score"]),
            float(run["log_loss"]),
            str(run["run_id"]),
        ),
    )
    return str(best["run_id"])


def select_best_baseline(val_results_path: str) -> str:
    """Selects the best baseline run from a validation results JSON.

    Args:
        val_results_path: Path to the results file written by the evaluation
            script for the validation cohort.

    Returns:
        The run ID of the selected baseline.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file was not computed on the validation cohort, has
            no runs, or a run lacks a ranking metric.
    """
    if not os.path.exists(val_results_path):
        raise FileNotFoundError(f"Validation results not found at: {val_results_path}")

    with open(val_results_path) as file:
        payload = json.load(file)

    if payload.get("cohort") != VALIDATION_COHORT:
        raise ValueError(
            f"{val_results_path} holds metrics for cohort {payload.get('cohort')!r}; "
            "selection reads validation metrics only."
        )
    return select_best_run(payload.get("individual_runs", []))


def select_backbone(layout: ResultsLayout, configuration: str | None = None) -> RunSpec:
    """Selects the baseline run to build transfer learning heads on.

    Reads the validation results of one baseline configuration, picks the best
    run with `select_best_run`, and returns that run's spec (found through its
    history). The spec gives the checkpoint path and the heads' `backbone_id`.

    Args:
        layout: The results layout to read from.
        configuration: The `config_id` of the baseline configuration to select
            within. Optional when exactly one baseline configuration has
            validation results.

    Returns:
        The spec of the selected baseline run.

    Raises:
        FileNotFoundError: If no baseline has validation results yet.
        ValueError: If several baseline configurations have validation results
            and none was named, if the results are not from the validation
            cohort, or if the selected run has no history.
    """
    if configuration is None:
        configurations = layout.configs_with_metrics(
            Paradigm.BASELINE, VALIDATION_COHORT
        )
        if not configurations:
            raise FileNotFoundError(
                f"No validation results for the baseline under {layout.root}. "
                "Run scripts/metrics/evaluate_baseline.py first."
            )
        if len(configurations) > 1:
            raise ValueError(
                "Several baseline configurations have validation results "
                f"({configurations}); name the configuration to select from."
            )
        configuration = configurations[0]

    selected = select_best_baseline(
        layout.metrics_path(Paradigm.BASELINE, configuration, VALIDATION_COHORT)
    )
    for history in load_runs(layout, Paradigm.BASELINE, configuration):
        if history.run_id == selected:
            return history.spec
    raise ValueError(
        f"The run {selected!r} selected on validation has no history under "
        f"baseline configuration {configuration!r}."
    )
