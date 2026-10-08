"""Visualization script for Classical Transfer Learning metrics on NIfTI data.

This script reads the per-configuration evaluation results written by
`scripts/metrics/evaluate_ctl.py` and, for every CTL configuration, produces
publication-quality charts: metric boxplots and comparative ROC curves over all
seeds, and the isolated ROC curve and confusion matrix of the run selected on the
validation cohort (never on test).
"""

import sys

from dementia_boost.core.identity import Paradigm
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.viz.run_plots import plot_paradigm


def main() -> None:
    """Generates all visualization artifacts for CTL NIfTI models."""
    logger = setup_logger("visualize_ctl_nifti")

    try:
        plotted = plot_paradigm(ResultsLayout(), Paradigm.CTL, logger)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to generate plots: {error}")
        sys.exit(1)

    if not plotted:
        logger.error("No CTL results found. Run evaluate_ctl.py first.")
        sys.exit(1)
    logger.info(f"Success! CTL plots generated for {len(plotted)} configuration(s).")


if __name__ == "__main__":
    main()
