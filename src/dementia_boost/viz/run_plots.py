"""Plotting steps shared by the visualization scripts, on the results layout.

`plot_paradigm` draws the evaluation plots of every configuration of a paradigm,
`plot_metric_comparisons` compares the metric distributions of all configurations
side by side, and `plot_losses` draws the loss curves from the saved histories.
Everything is
found through the layout and the stored specs, never through file names. The
drawing itself is done by `MetricsVisualizer`.

The run highlighted in the isolated ROC curve and the confusion matrix is chosen
on the validation results, like every other selection; the test results are only
drawn.
"""

import json
import logging
import os
from logging import Logger

from dementia_boost.core.identity import Paradigm, label
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.run_listing import group_by_config, load_runs
from dementia_boost.telemetry.selection import select_best_validation_run
from dementia_boost.viz.display_names import production_labels
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


def plot_metric_comparisons(
    layout: ResultsLayout, logger: Logger | None = None, production: bool = False
) -> dict[str, list[str]]:
    """Compares every configuration's metric distributions on each cohort.

    For the test and the validation cohort it writes one figure with a box per
    configuration and metric to `<root>/plots`, from the results of every
    configuration of every paradigm that has them.

    Args:
        layout: The results layout to read metrics from and write plots to.
        logger: Logger for progress. Defaults to a module logger.
        production: Whether to use the short presentation names for the
            configurations and metrics (see `display_names`) and write
            `metric_comparison_<cohort>_production.png`, so the figure with the
            full configuration labels is not overwritten.

    Returns:
        Mapping of each cohort (`test`, `val`) to the labels of the
        configurations drawn on it. Cohorts without results are omitted.
    """
    logger = logger or _DEFAULT_LOGGER
    drawn: dict[str, list[str]] = {}

    for cohort in ("test", "val"):
        found: dict[str, tuple[str, str]] = {}
        shown: dict[str, tuple[str, str | None]] = {}
        for paradigm in Paradigm:
            for config in layout.configs_with_metrics(paradigm, cohort):
                path = layout.metrics_path(paradigm, config, cohort)
                with open(path) as file:
                    configuration = json.load(file)["configuration"]
                found[config] = (configuration["label"], path)
                shown[config] = (paradigm.value, configuration["spec"].get("ansatz"))
        if not found:
            continue

        names = production_labels(shown) if production else None
        paths = {
            names[config] if names else full_label: path
            for config, (full_label, path) in found.items()
        }
        name = "validation" if cohort == "val" else cohort
        suffix = "_production" if production else ""
        logger.info(f"Comparing {len(paths)} configuration(s) on the {name} cohort")
        MetricsVisualizer(
            output_dir=os.path.join(layout.root, "plots")
        ).plot_metric_comparison(
            paths,
            name,
            f"metric_comparison_{name}{suffix}.png",
            production=production,
        )
        drawn[cohort] = list(paths)

    return drawn


def plot_losses(
    layout: ResultsLayout, logger: Logger | None = None, production: bool = False
) -> dict[str, str]:
    """Draws the loss plots of every configuration from the saved histories.

    Per configuration it writes one loss curve per run and a multi-seed loss
    distribution to the configuration's plot directory, and finally the
    cross-configuration validation-loss comparison to `<root>/plots`.

    Args:
        layout: The results layout to read histories from and write plots to.
        logger: Logger for progress. Defaults to a module logger.
        production: Whether to draw only the cross-configuration comparison,
            with the short presentation names (see `display_names`), to
            `loss_comparison_production.png`. The per-configuration plots and
            the comparison with the full labels are not touched.

    Returns:
        Mapping of each configuration's label to its history directory. Empty if
        there are no histories.
    """
    logger = logger or _DEFAULT_LOGGER
    available: dict[str, str] = {}
    shown: dict[str, tuple[str, str | None]] = {}
    directories: dict[str, str] = {}

    for config, histories in group_by_config(load_runs(layout)).items():
        first = histories[0].spec
        paradigm = first.paradigm
        history_dir = layout.history_dir(paradigm, config)

        if not production:
            visualizer = MetricsVisualizer(
                output_dir=layout.plots_dir(paradigm, config)
            )
            logger.info(
                f"Plotting {len(histories)} loss curve(s) for {label(first, False)}"
            )
            for history in histories:
                visualizer.plot_loss_curve(
                    layout.history_path(history.spec), prefix=paradigm.value
                )
            visualizer.plot_loss_distribution(history_dir, prefix=paradigm.value)
        available[label(first, include_seed=False)] = history_dir
        shown[config] = (paradigm.value, first.ansatz)
        directories[config] = history_dir

    if production:
        names = production_labels(shown)
        available = {names[config]: path for config, path in directories.items()}
    if available:
        MetricsVisualizer(
            output_dir=os.path.join(layout.root, "plots")
        ).plot_loss_comparison(
            available,
            filename="loss_comparison_production.png"
            if production
            else "loss_comparison.png",
        )
    return available
