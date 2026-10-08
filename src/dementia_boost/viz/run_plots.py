"""Plotting steps shared by the visualization scripts, on the results layout.

`plot_paradigm` draws the evaluation plots of every configuration of a paradigm,
and `plot_losses` draws the loss curves from the saved histories. Everything is
found through the layout and the stored specs, never through file names. The
drawing itself is done by `MetricsVisualizer`.

The run highlighted in the isolated ROC curve and the confusion matrix is chosen
on the validation results, like every other selection; the test results are only
drawn.
"""

import logging
import os
from logging import Logger

from dementia_boost.core.identity import Paradigm, label
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.run_listing import group_by_config, load_runs
from dementia_boost.telemetry.selection import select_best_validation_run
from dementia_boost.viz.visualizer import MetricsVisualizer

_DEFAULT_LOGGER = logging.getLogger(__name__)


def plot_paradigm(
    layout: ResultsLayout,
    paradigm: Paradigm | str,
    logger: Logger | None = None,
    configuration: str | None = None,
) -> list[str]:
    """Draws the evaluation plots of every configuration of a paradigm.

    For each configuration with test results it writes the metric distributions
    and the comparative ROC curves over all seeds, and an isolated ROC curve and
    confusion matrix for the run that is best on the validation results.

    Args:
        layout: The results layout to read from and write plots to.
        paradigm: The paradigm to plot.
        logger: Logger for progress. Defaults to a module logger.
        configuration: Restrict to one `config_id`. Defaults to all.

    Returns:
        The `config_id`s that were plotted. Empty if there are no results.

    Raises:
        FileNotFoundError: If a configuration has test results but no
            validation results, because the highlighted run cannot then be
            chosen without looking at the test cohort.
    """
    logger = logger or _DEFAULT_LOGGER
    paradigm = Paradigm(paradigm)
    configurations = (
        [configuration]
        if configuration is not None
        else layout.configs_with_metrics(paradigm, "test")
    )

    plotted: list[str] = []
    for config in configurations:
        test_path = layout.metrics_path(paradigm, config, "test")
        selected = select_best_validation_run(
            layout.metrics_path(paradigm, config, "val")
        )
        visualizer = MetricsVisualizer(output_dir=layout.plots_dir(paradigm, config))
        prefix = paradigm.value

        logger.info(
            f"Plotting configuration {config}; run selected on validation: {selected}"
        )
        visualizer.plot_metric_distributions(test_path, prefix=prefix)
        visualizer.plot_comparative_roc(test_path, prefix=prefix)
        visualizer.plot_isolated_roc(test_path, run_id=selected, prefix=prefix)
        visualizer.plot_confusion_matrix(test_path, run_id=selected, prefix=prefix)
        plotted.append(config)

    return plotted


def plot_losses(layout: ResultsLayout, logger: Logger | None = None) -> dict[str, str]:
    """Draws the loss plots of every configuration from the saved histories.

    Per configuration it writes one loss curve per run and a multi-seed loss
    distribution to the configuration's plot directory, and finally the
    cross-configuration validation-loss comparison to `<root>/plots`.

    Args:
        layout: The results layout to read histories from and write plots to.
        logger: Logger for progress. Defaults to a module logger.

    Returns:
        Mapping of each configuration's label to its history directory. Empty if
        there are no histories.
    """
    logger = logger or _DEFAULT_LOGGER
    available: dict[str, str] = {}

    for config, histories in group_by_config(load_runs(layout)).items():
        first = histories[0].spec
        paradigm = first.paradigm
        history_dir = layout.history_dir(paradigm, config)
        visualizer = MetricsVisualizer(output_dir=layout.plots_dir(paradigm, config))

        logger.info(
            f"Plotting {len(histories)} loss curve(s) for {label(first, False)}"
        )
        for history in histories:
            visualizer.plot_loss_curve(
                layout.history_path(history.spec), prefix=paradigm.value
            )
        visualizer.plot_loss_distribution(history_dir, prefix=paradigm.value)
        available[label(first, include_seed=False)] = history_dir

    if available:
        MetricsVisualizer(
            output_dir=os.path.join(layout.root, "plots")
        ).plot_loss_comparison(available)
    return available
