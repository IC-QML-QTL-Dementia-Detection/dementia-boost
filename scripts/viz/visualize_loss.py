"""Visualization script regenerating every loss-curve plot from saved histories.

The training scripts only save raw per-epoch histories as JSON. This script is
the only place those histories are turned into plots. It reads the histories of
every configuration through `ResultsLayout` and writes, per configuration, one
loss-curve plot per run and one multi-seed loss distribution plot to that
configuration's plot directory, plus the cross-configuration validation-loss
comparison to `data/results/plots/loss_comparison.png`. Configurations without
any saved history simply do not appear, so it can run while sweeps are in
progress.
"""

import sys

import matplotlib

from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.viz.run_plots import plot_losses


def main() -> None:
    """Generates all loss-curve visualization artifacts from saved histories."""
    matplotlib.use("Agg")
    logger = setup_logger("visualize_loss_nifti")

    try:
        available = plot_losses(ResultsLayout(), logger)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to generate plots: {error}")
        sys.exit(1)

    if not available:
        logger.error("No training histories found. Run a training script first.")
        sys.exit(1)
    logger.info(f"Success! Loss visualizations generated for {list(available)}.")


if __name__ == "__main__":
    main()
