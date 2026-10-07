"""Visualization script for Qiskit Quantum Transfer Learning metrics on NIfTI data.

This script reads the per-configuration evaluation results written by
`scripts/metrics/evaluate_qiskit_qtl.py` and, for every Qiskit QTL configuration,
produces publication-quality charts: metric boxplots and comparative ROC curves
over all seeds, and the isolated ROC curve and confusion matrix of the run
selected on the validation cohort (never on test).
"""

import sys

from dementia_boost.core.identity import Paradigm
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.viz.run_plots import plot_paradigm


def main() -> None:
    """Generates all visualization artifacts for Qiskit QTL NIfTI models."""
    logger = setup_logger("visualize_qiskit_qtl_nifti")

    try:
        plotted = plot_paradigm(ResultsLayout(), Paradigm.QISKIT_QTL, logger)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to generate plots: {error}")
        sys.exit(1)

    if not plotted:
        logger.error("No Qiskit QTL results found. Run evaluate_qiskit_qtl.py first.")
        sys.exit(1)
    logger.info(
        f"Success! Qiskit QTL plots generated for {len(plotted)} configuration(s)."
    )


if __name__ == "__main__":
    main()
