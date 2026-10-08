"""Tests for the per-paradigm metric plots and the loss plots on the results layout.

Regression coverage
-------------------
- The isolated ROC curve and confusion matrix were drawn for the run with the
  best test accuracy and AUC, so a run was chosen on the test cohort.
- Loss plots read every `*.json` in a history directory, which now includes the
  configuration file.
"""

from pathlib import Path

import numpy as np
from conftest import build_spec

from dementia_boost.core.identity import Paradigm, RunSpec, config_id, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.training.checkpoint_evaluation import save_configuration_results
from dementia_boost.viz.run_plots import plot_losses, plot_paradigm

_LABELS = np.array([0, 1] * 10)
_PERFECT = np.where(_LABELS == 1, 0.9, 0.1)
_WORSE_THAN_CHANCE = np.where(_LABELS == 1, 0.2, 0.8)


def _write_results(layout: ResultsLayout, spec_a: RunSpec, spec_b: RunSpec) -> None:
    """Writes results where run A is best on test and run B is best on validation."""
    metrics = MetricsAnalyzer.calculate_metrics
    results = {
        "val": [
            metrics(run_id(spec_a), _LABELS, _WORSE_THAN_CHANCE),
            metrics(run_id(spec_b), _LABELS, _PERFECT),
        ],
        "test": [
            metrics(run_id(spec_a), _LABELS, _PERFECT),
            metrics(run_id(spec_b), _LABELS, _WORSE_THAN_CHANCE),
        ],
    }
    save_configuration_results(layout, spec_a, results)


class TestPlotParadigm:
    """Validates the plots written for each configuration of a paradigm."""

    def test_isolated_plots_use_the_run_selected_on_validation(
        self, tmp_path: Path
    ) -> None:
        """The highlighted run is the best on validation, not the best on test."""
        layout = ResultsLayout(str(tmp_path))
        spec_a, spec_b = build_spec("pl_qtl", seed=1), build_spec("pl_qtl", seed=2)
        _write_results(layout, spec_a, spec_b)

        plot_paradigm(layout, Paradigm.PL_QTL)

        plots = Path(layout.plots_dir("pl_qtl", config_id(spec_a)))
        names = {path.name for path in plots.glob("*.png")}
        assert f"pl_qtl_isolated_roc_{run_id(spec_b)}.png" in names
        assert f"pl_qtl_cm_{run_id(spec_b)}.png" in names
        assert not any(run_id(spec_a) in name for name in names)

    def test_writes_the_distribution_and_comparative_roc_plots(
        self, tmp_path: Path
    ) -> None:
        """The all-seeds plots are written next to the isolated ones."""
        layout = ResultsLayout(str(tmp_path))
        spec_a, spec_b = build_spec("pl_qtl", seed=1), build_spec("pl_qtl", seed=2)
        _write_results(layout, spec_a, spec_b)

        plot_paradigm(layout, Paradigm.PL_QTL)

        plots = Path(layout.plots_dir("pl_qtl", config_id(spec_a)))
        for name in ("pl_qtl_distributions.png", "pl_qtl_roc_curves.png"):
            assert (plots / name).stat().st_size > 0

    def test_returns_the_configurations_it_plotted(self, tmp_path: Path) -> None:
        """The caller learns which configurations got plots."""
        layout = ResultsLayout(str(tmp_path))
        spec_a, spec_b = build_spec("pl_qtl", seed=1), build_spec("pl_qtl", seed=2)
        _write_results(layout, spec_a, spec_b)

        assert plot_paradigm(layout, Paradigm.PL_QTL) == [config_id(spec_a)]

    def test_nothing_to_plot_returns_nothing(self, tmp_path: Path) -> None:
        """A paradigm without results produces no plots."""
        assert plot_paradigm(ResultsLayout(str(tmp_path)), Paradigm.CTL) == []


class TestPlotLosses:
    """Validates the loss plots read from the histories on disk."""

    def _write_history(self, layout: ResultsLayout, spec: RunSpec) -> None:
        """Writes a short history for a spec."""
        path = Path(layout.history_path(spec))
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs = [
            EpochRecord(e, 0.7 - 0.05 * e, 0.5, 0.72 - 0.04 * e, 0.5, spec.lr, 1.0)
            for e in range(1, 5)
        ]
        MetricsAnalyzer.save_history(
            TrainingHistory(spec=spec, epochs=epochs, extras={}), str(path)
        )

    def test_plots_each_configuration_and_the_comparison(self, tmp_path: Path) -> None:
        """Per-run curves and a distribution go to the configuration's plot
        directory, and the cross-configuration comparison to the plots root. The
        configuration file in the history directory is not read as a history."""
        layout = ResultsLayout(str(tmp_path))
        specs = [
            build_spec("pl_qtl", seed=1),
            build_spec("pl_qtl", seed=2),
            build_spec("ctl", seed=1),
        ]
        for spec in specs:
            self._write_history(layout, spec)
            layout.write_config(spec)

        plotted = plot_losses(layout)

        qtl_plots = Path(layout.plots_dir("pl_qtl", config_id(specs[0])))
        for spec in specs[:2]:
            assert (qtl_plots / f"pl_qtl_{run_id(spec)}_loss.png").stat().st_size > 0
        assert (qtl_plots / "pl_qtl_loss_distribution.png").stat().st_size > 0
        assert (Path(layout.root) / "plots" / "loss_comparison.png").stat().st_size > 0
        assert len(plotted) == 2

    def test_no_histories_plots_nothing(self, tmp_path: Path) -> None:
        """An empty layout gives an empty result instead of blank figures."""
        assert plot_losses(ResultsLayout(str(tmp_path))) == {}
