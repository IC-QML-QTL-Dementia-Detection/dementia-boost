"""Plotting: publication-ready figures from saved results and histories.

`MetricsVisualizer` draws the individual figures (ROC curves, metric
distributions, confusion matrices, loss curves). The functions in `run_plots`
find their inputs through the results layout and write the figures next to them.
The scripts under `scripts/viz` are thin callers of this package.
"""

from .run_plots import plot_losses, plot_paradigm
from .visualizer import MetricsVisualizer

__all__ = [
    "MetricsVisualizer",
    "plot_losses",
    "plot_paradigm",
]
