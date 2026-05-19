import os
import sys

from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.visualizer import MetricsVisualizer


def main() -> None:
    logger = setup_logger("visualize_baseline")

    results_json_path = "./data/results/metrics/baseline_results.json"
    plots_output_dir = "./data/results/plots"

    optimal_run_id = "baseline_seed_2"

    if not os.path.exists(results_json_path):
        logger.error(f"Results file not found at: {results_json_path}")
        logger.error("Please run evaluate_baselines.py first.")
        sys.exit(1)

    logger.info(f"Initializing Visualizer. Output directory: {plots_output_dir}")
    visualizer = MetricsVisualizer(output_dir=plots_output_dir)

    try:
        logger.info("Generating Metric Distributions Box Plot...")
        visualizer.plot_metric_distributions(results_json_path, prefix="baseline")

        logger.info("Generating Comparative ROC Curves...")
        visualizer.plot_comparative_roc(results_json_path, prefix="baseline")

        logger.info(f"Generating Isolated ROC Curve for {optimal_run_id}...")
        visualizer.plot_isolated_roc(
            results_json_path,
            run_id=optimal_run_id,
            prefix="baseline",
        )

        logger.info(f"Generating Confusion Matrix Heatmap for {optimal_run_id}...")
        visualizer.plot_confusion_matrix(
            results_json_path,
            run_id=optimal_run_id,
            prefix="baseline",
        )

        logger.info("Success! All baseline visualizations have been generated.")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
