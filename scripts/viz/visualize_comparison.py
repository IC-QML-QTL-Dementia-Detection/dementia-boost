"""Visualization script comparing the metric distributions of all configurations.

This script reads the evaluation results of every configuration of every
paradigm (written by the `evaluate_*` scripts) and draws, for the test cohort and
for the validation cohort, one figure with a box per configuration and metric to
`data/results/plots/metric_comparison_<cohort>.png`. Use it to compare, for
example, the embeddings of the Qiskit quantum head against the baseline, CTL, and
PennyLane references. Configurations without results simply do not appear.
"""

import sys

import matplotlib

from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.viz.run_plots import plot_metric_comparisons


def main() -> None:
    """Generates the cross-configuration metric comparison figures."""
    matplotlib.use("Agg")
    logger = setup_logger("visualize_comparison_nifti")

    try:
        drawn = plot_metric_comparisons(ResultsLayout(), logger)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to generate plots: {error}")
        sys.exit(1)

    if not drawn:
        logger.error("No evaluation results found. Run the evaluate_* scripts first.")
        sys.exit(1)
    logger.info(f"Success! Comparison figures generated for cohorts {list(drawn)}.")


if __name__ == "__main__":
    main()
