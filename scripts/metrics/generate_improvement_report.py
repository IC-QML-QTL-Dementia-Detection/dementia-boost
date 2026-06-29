import json
import os

from dementia_boost.telemetry.logger import setup_logger


def calculate_percentage_delta(base: float, new: float) -> float:
    """Computes the relative percentage improvement."""
    if base == 0.0:
        return 0.0
    return ((new - base) / base) * 100.0


def get_best_run(runs: list[dict], sort_metric: str = "accuracy") -> dict:
    """Isolates the best-performing model runs."""
    return max(runs, key=lambda x: x[sort_metric])


def main() -> None:
    logger = setup_logger("imp_report")

    baseline_path = "./data/results/metrics/baseline_results.json"
    tl_path = "./data/results/metrics/tl_results.json"
    output_path = "./data/results/metrics/comparative_report.json"

    if not os.path.exists(baseline_path):
        logger.error(f"Missing baseline telemetry at: {baseline_path}")
        return

    with open(baseline_path) as f:
        base_data = json.load(f)

    if not os.path.exists(tl_path):
        logger.error(f"Missing Transfer Learning telemetry at: {tl_path}")
        return

    with open(tl_path) as f:
        tl_data = json.load(f)

    best_base = get_best_run(base_data["individual_runs"], "accuracy")
    best_tl = get_best_run(tl_data["individual_runs"], "accuracy")

    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]

    report = {
        "metadata": {
            "best_baseline_run": best_base["run_id"],
            "best_ctl_run": best_tl["run_id"],
            "best_qtl_run": "pending",
        },
        "comparisons": {},
    }

    for m in metrics:
        base_val = best_base[m]
        tl_val = best_tl[m]

        report["comparisons"][m] = {
            "baseline": base_val,
            "ctl": tl_val,
            "ctl_imp_base_pct": calculate_percentage_delta(base_val, tl_val),
            "qtl": 0.0,
            "qtl_imp_base_pct": 0.0,
            "qtl_imp_ctl_pct": 0.0,
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Report generated successfully at: {output_path}")
    logger.info(
        f"Optimal Baseline: {best_base['run_id']} | Optimal CTL: {best_tl['run_id']}"
    )


if __name__ == "__main__":
    main()
