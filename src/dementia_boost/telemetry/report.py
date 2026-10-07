"""Comparative report across paradigms.

The report leads with the mean and standard deviation of every metric across the
seeds of a paradigm, measured on the test cohort. Next to them it shows the test
metrics of the run that was selected on the validation cohort. No run is chosen
on test metrics.
"""

from collections.abc import Mapping
from typing import Any

from dementia_boost.telemetry.selection import select_best_run

REPORT_METRICS = ("accuracy", "precision", "recall", "f1_score", "auc")
SELECTION_RULE = "validation AUC-ROC, then F1-score, then lowest log loss"
DELTA_PAIRS = (
    ("ctl", "baseline"),
    ("qtl", "baseline"),
    ("qtl", "ctl"),
    ("qiskit_qtl", "baseline"),
    ("qiskit_qtl", "ctl"),
    ("qiskit_qtl", "qtl"),
)


def calculate_percentage_delta(base: float, new: float) -> float:
    """Computes the relative percentage change between two scalar metrics.

    Args:
        base: Reference metric score.
        new: Compared metric score.

    Returns:
        The percentage change from base to new. Returns 0.0 if base is zero.
    """
    if base == 0.0:
        return 0.0
    return ((new - base) / base) * 100.0


def build_comparative_report(
    test_results: Mapping[str, Mapping[str, Any]],
    val_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Builds the comparison of paradigms from their saved results.

    Args:
        test_results: Mapping of paradigm name ("baseline", "ctl", "qtl",
            "qiskit_qtl") to its test-cohort results payload.
        val_results: Mapping of the same paradigm names to their
            validation-cohort results payloads. Used only to select a run.

    Returns:
        A dictionary with `metadata` (the selection rule and the run selected
        per paradigm) and `comparisons`. For each metric, `comparisons` holds
        `paradigms` (per paradigm: `mean` and `std` across seeds on test, and
        `selected_on_validation`, the test value of the selected run) and
        `delta_pct_of_means` (percentage change of the means, for every pair
        with both paradigms present).

    Raises:
        ValueError: If there are no paradigms, validation and test cover
            different paradigms, a payload is marked with the wrong cohort, or a
            run selected on validation is absent from the test results.
    """
    if not test_results:
        raise ValueError("No paradigms to report.")
    if set(test_results) != set(val_results):
        raise ValueError(
            "Validation and test results cover different paradigms: "
            f"{sorted(val_results)} versus {sorted(test_results)}."
        )
    for name, payload in test_results.items():
        _require_cohort(name, payload, "test")
    for name, payload in val_results.items():
        _require_cohort(name, payload, "val")

    selected_runs = {
        name: select_best_run(val_results[name]["individual_runs"])
        for name in test_results
    }
    selected_metrics = {
        name: _run_metrics(name, test_results[name], selected_runs[name])
        for name in test_results
    }

    comparisons: dict[str, Any] = {}
    for metric in REPORT_METRICS:
        paradigms = {
            name: {
                "mean": float(payload["aggregated_statistics"][metric]["mean"]),
                "std": float(payload["aggregated_statistics"][metric]["std"]),
                "selected_on_validation": float(selected_metrics[name][metric]),
            }
            for name, payload in test_results.items()
        }
        comparisons[metric] = {
            "paradigms": paradigms,
            "delta_pct_of_means": {
                f"{new}_vs_{base}": calculate_percentage_delta(
                    paradigms[base]["mean"], paradigms[new]["mean"]
                )
                for new, base in DELTA_PAIRS
                if new in paradigms and base in paradigms
            },
        }

    return {
        "metadata": {"selection": SELECTION_RULE, "selected_runs": selected_runs},
        "comparisons": comparisons,
    }


def _require_cohort(name: str, payload: Mapping[str, Any], expected: str) -> None:
    """Raises if a results payload was not computed on the expected cohort.

    Args:
        name: Paradigm name, for the error message.
        payload: Results payload with a `cohort` marker.
        expected: The cohort the payload must carry.

    Raises:
        ValueError: If the marker is missing or different.
    """
    if payload.get("cohort") != expected:
        raise ValueError(
            f"{name}: expected a {expected!r} cohort payload, "
            f"got cohort {payload.get('cohort')!r}."
        )


def _run_metrics(
    name: str, payload: Mapping[str, Any], run_id: str
) -> Mapping[str, Any]:
    """Finds one run's metrics in a results payload.

    Args:
        name: Paradigm name, for the error message.
        payload: Results payload with `individual_runs`.
        run_id: The run to find.

    Returns:
        The metrics entry of the run.

    Raises:
        ValueError: If the payload has no such run.
    """
    for run in payload["individual_runs"]:
        if run["run_id"] == run_id:
            return run
    raise ValueError(
        f"{name}: run {run_id!r} selected on validation is not in the test results."
    )
