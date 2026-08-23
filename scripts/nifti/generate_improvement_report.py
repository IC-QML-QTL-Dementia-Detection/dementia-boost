"""Comparative improvement report generator across paradigms on NIfTI data.

This script aggregates evaluation results across the Classical Baseline, Classical
Transfer Learning (CTL), and Quantum Transfer Learning (QTL) runs on NIfTI data.
It identifies the top-performing run per paradigm using composite metric ranking,
calculates relative percentage improvements, and exports a comparative JSON report.
"""

import json
import os

from dementia_boost.telemetry.logger import setup_logger


def calculate_percentage_delta(base: float, new: float) -> float:
    """Computes the relative percentage improvement between two scalar metrics.

    Args:
        base: Baseline metric score.
        new: Updated or transfer learning metric score.

    Returns:
        The percentage change from base to new. Returns 0.0 if base is zero.
    """
    if base == 0.0:
        return 0.0
    return ((new - base) / base) * 100.0


def get_best_run(runs: list[dict[str, float | str]]) -> dict[str, float | str]:
    """Isolates the best-performing model run using composite multi-metric ranking.

    Ranks runs primarily by Accuracy, followed by F1-score and AUC-ROC to prevent
    selecting runs overfitted to a single metric.

    Args:
        runs: List of individual run dictionaries from evaluation JSON.

    Returns:
        The dictionary of the top-ranked model run.
    """
    return max(
        runs,
        key=lambda item: (
            float(item.get("accuracy", 0.0)),
            float(item.get("f1_score", 0.0)),
            float(item.get("auc", 0.0)),
        ),
    )


def main() -> None:
    """Generates the quantitative comparative improvement report for NIfTI."""
    logger = setup_logger("improvement_report_nifti")

    baseline_path = "./data/results/metrics/nifti/baseline_results.json"
    ctl_path = "./data/results/metrics/nifti/tl_results.json"
    qtl_path = "./data/results/metrics/nifti/qtl_results.json"
    output_path = "./data/results/metrics/nifti/comparative_report.json"

    if not os.path.exists(baseline_path):
        logger.error(f"Missing baseline telemetry at: {baseline_path}")
        return

    with open(baseline_path) as file:
        base_data = json.load(file)

    if not os.path.exists(ctl_path):
        logger.error(f"Missing Classical Transfer Learning telemetry at: {ctl_path}")
        return

    with open(ctl_path) as file:
        ctl_data = json.load(file)

    qtl_data = None
    if os.path.exists(qtl_path):
        with open(qtl_path) as file:
            qtl_data = json.load(file)

    best_base = get_best_run(base_data["individual_runs"])
    best_ctl = get_best_run(ctl_data["individual_runs"])

    best_qtl = None
    best_qtl_run_id = "pending"
    if qtl_data and qtl_data.get("individual_runs"):
        best_qtl = get_best_run(qtl_data["individual_runs"])
        best_qtl_run_id = str(best_qtl["run_id"])

    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]

    report: dict[str, dict[str, str | dict[str, float]]] = {
        "metadata": {
            "best_baseline_run": str(best_base["run_id"]),
            "best_ctl_run": str(best_ctl["run_id"]),
            "best_qtl_run": best_qtl_run_id,
        },
        "comparisons": {},
    }

    for metric in metrics:
        base_val = float(best_base[metric])
        ctl_val = float(best_ctl[metric])
        qtl_val = float(best_qtl[metric]) if best_qtl else 0.0

        report["comparisons"][metric] = {
            "baseline": base_val,
            "ctl": ctl_val,
            "ctl_imp_base_pct": calculate_percentage_delta(base_val, ctl_val),
            "qtl": qtl_val,
            "qtl_imp_base_pct": (
                calculate_percentage_delta(base_val, qtl_val) if best_qtl else 0.0
            ),
            "qtl_imp_ctl_pct": (
                calculate_percentage_delta(ctl_val, qtl_val) if best_qtl else 0.0
            ),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)

    logger.info(f"Report generated successfully at: {output_path}")
    logger.info(
        f"Optimal Baseline: {best_base['run_id']} | Optimal CTL: {best_ctl['run_id']} "
        f"| Optimal QTL: {best_qtl_run_id}"
    )


if __name__ == "__main__":
    main()
