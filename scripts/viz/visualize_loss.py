"""Visualization script regenerating every loss-curve plot from saved histories.

The training scripts only save raw per-epoch histories as JSON. This script is
the only place those histories are turned into plots. It scans the per-paradigm
history directories under `data/results/histories/nifti` and writes:

- `<loss root>/<paradigm>/`: one loss-curve plot per run and one multi-seed loss
  distribution plot for that paradigm.
- `<loss root>/loss_comparison.png`: the cross-paradigm validation-loss
  comparison.

The loss root is `data/results/plots/nifti/loss`. Paradigms without any saved
history are skipped, so it can run while only some sweeps have finished.
"""

import sys
from pathlib import Path

import matplotlib

from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.visualizer import MetricsVisualizer

DEFAULT_HISTORIES_ROOT: str = "./data/results/histories/nifti"
DEFAULT_LOSS_PLOTS_ROOT: str = "./data/results/plots/nifti/loss"
PARADIGM_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "ctl": "CTL",
    "qtl": "PL-QTL",
    "qiskit_qtl": "Qiskit-QTL",
}


def main() -> None:
    """Generates all loss-curve visualization artifacts from saved histories."""
    matplotlib.use("Agg")
    logger = setup_logger("visualize_loss_nifti")

    logger.info(f"Loss plots root directory: {DEFAULT_LOSS_PLOTS_ROOT}")

    try:
        available: dict[str, str] = {}
        for paradigm, label in PARADIGM_LABELS.items():
            history_dir = Path(DEFAULT_HISTORIES_ROOT) / paradigm
            history_files = sorted(history_dir.glob("*.json"))

            if not history_files:
                logger.warning(
                    f"No histories found for '{paradigm}' in {history_dir}. Skipping."
                )
                continue

            paradigm_plots_dir = Path(DEFAULT_LOSS_PLOTS_ROOT) / paradigm
            paradigm_visualizer = MetricsVisualizer(output_dir=str(paradigm_plots_dir))

            logger.info(
                f"Generating {len(history_files)} loss curves for {label} "
                f"in {paradigm_plots_dir}..."
            )
            for history_file in history_files:
                paradigm_visualizer.plot_loss_curve(str(history_file), prefix=paradigm)

            logger.info(f"Generating loss distribution for {label}...")
            paradigm_visualizer.plot_loss_distribution(
                str(history_dir), prefix=paradigm
            )
            available[label] = str(history_dir)

        if not available:
            logger.error(f"No training histories found under {DEFAULT_HISTORIES_ROOT}.")
            logger.error("Please execute a training script under scripts/training/.")
            sys.exit(1)

        logger.info(f"Generating cross-paradigm comparison of {list(available)}...")
        MetricsVisualizer(output_dir=DEFAULT_LOSS_PLOTS_ROOT).plot_loss_comparison(
            available
        )

        logger.info("Success! All loss visualizations have been generated.")
    except Exception as error:
        logger.error(f"Failed to generate plots: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
