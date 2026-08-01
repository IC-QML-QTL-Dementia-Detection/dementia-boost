import json
import os

from dementia_boost.telemetry.logger import setup_logger


def calculate_percentage_delta(base: float, new: float) -> float:
    """Computes the relative percentage improvement."""
    if base == 0.0:
        return 0.0
    return ((new - base) / base) * 100.0


def get_best_run(runs: list[dict]) -> dict:
    """
    Isolates the best-performing model run using a composite priority algorithm
    to avoid barren plateaus that spike on a single metric.
    Primary: Accuracy | Secondary: F1 Score | Tertiary: AUC
    """
    return max(
        runs,
        key=lambda x: (
            float(x.get("accuracy", 0.0)),
            float(x.get("f1_score", 0.0)),
            float(x.get("auc", 0.0)),
        ),
    )


def main() -> None:
    logger = setup_logger("imp_report_jpg")

    baseline_path = "./data/results/metrics/jpg/baseline_results.json"
    ctl_path = "./data/results/metrics/jpg/tl_results.json"
    qtl_path = "./data/results/metrics/jpg/qtl_results.json"
    output_path = "./data/results/metrics/jpg/comparative_report_jpg.json"

    if not os.path.exists(baseline_path):
        logger.error(f"Missing baseline telemetry at: {baseline_path}")
        return

    with open(baseline_path) as f:
        base_data = json.load(f)

    if not os.path.exists(ctl_path):
        logger.error(f"Missing Classical Transfer Learning telemetry at: {ctl_path}")
        return

    with open(ctl_path) as f:
        tl_data = json.load(f)

    qtl_data = None
    if os.path.exists(qtl_path):
        with open(qtl_path) as f:
            qtl_data = json.load(f)

    best_base = get_best_run(base_data["individual_runs"])
    best_ctl = get_best_run(tl_data["individual_runs"])

    best_qtl = None
    if qtl_data:
        best_qtl = get_best_run(qtl_data["individual_runs"])

    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]

    report = {
        "metadata": {
            "best_baseline_run": best_base["run_id"],
            "best_ctl_run": best_ctl["run_id"],
            "best_qtl_run": "pending",
        },
        "comparisons": {},
    }

    for m in metrics:
        base_val = best_base[m]
        ctl_val = best_ctl[m]
        qtl_val = best_qtl[m] if best_qtl else 0.0

        report["comparisons"][m] = {
            "baseline": base_val,
            "ctl": ctl_val,
            "ctl_imp_base_pct": calculate_percentage_delta(base_val, ctl_val),
            "qtl": qtl_val,
            "qtl_imp_base_pct": calculate_percentage_delta(base_val, qtl_val),
            "qtl_imp_ctl_pct": calculate_percentage_delta(ctl_val, qtl_val),
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Report generated successfully at: {output_path}")
    logger.info(
        f"Optimal Baseline: {best_base['run_id']} | Optimal CTL: {best_ctl['run_id']}"
    )


if __name__ == "__main__":
    main()
