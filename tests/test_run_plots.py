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
import pytest
from conftest import build_spec

from dementia_boost.core.identity import Paradigm, RunSpec, config_id, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.training.checkpoint_evaluation import save_configuration_results
from dementia_boost.viz.run_plots import (
    plot_losses,
    plot_metric_comparisons,
    plot_paradigm,
)
from dementia_boost.viz.visualizer import MetricsVisualizer

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

    def test_writes_every_plot_with_the_vector_formats(self, tmp_path: Path) -> None:
        """The all-seeds and isolated plots are written, every one also in the
        extra formats with the PNG's name, and the caller learns which
        configurations got plots."""
        layout = ResultsLayout(str(tmp_path))
        spec_a, spec_b = build_spec("pl_qtl", seed=1), build_spec("pl_qtl", seed=2)
        _write_results(layout, spec_a, spec_b)

        plotted = plot_paradigm(layout, Paradigm.PL_QTL, vector_formats=("svg", "pdf"))

        assert plotted == [config_id(spec_a)]
        plots = Path(layout.plots_dir("pl_qtl", config_id(spec_a)))
        stems = {path.stem for path in plots.glob("*.png")}
        assert stems == {
            "pl_qtl_distributions",
            "pl_qtl_roc_curves",
            f"pl_qtl_isolated_roc_{run_id(spec_b)}",
            f"pl_qtl_cm_{run_id(spec_b)}",
        }
        assert all(path.stat().st_size > 0 for path in plots.glob("*.png"))
        for extension in ("svg", "pdf"):
            assert {path.stem for path in plots.glob(f"*.{extension}")} == stems

    def test_nothing_to_plot_returns_nothing(self, tmp_path: Path) -> None:
        """A paradigm without results produces no plots."""
        assert plot_paradigm(ResultsLayout(str(tmp_path)), Paradigm.CTL) == []


class TestPlotLosses:
    """Validates the loss plots read from the histories on disk."""

    def _write_history(self, layout: ResultsLayout, spec: RunSpec) -> None:
        """Writes a short history for a spec, validated on even epochs only.

        Seed 1 stops after 4 epochs and the other seeds run 6, so the plots must
        tolerate missing validation values and runs of unequal length.
        """
        path = Path(layout.history_path(spec))
        path.parent.mkdir(parents=True, exist_ok=True)
        n_epochs = 4 if spec.seed == 1 else 6
        epochs = [
            EpochRecord(
                epoch=e,
                train_loss=0.7 - 0.05 * e,
                train_acc=0.5,
                val_loss=0.72 - 0.04 * e if e % 2 == 0 else None,
                val_acc=0.5 if e % 2 == 0 else None,
                lr=spec.lr,
                duration_s=1.0,
            )
            for e in range(1, n_epochs + 1)
        ]
        MetricsAnalyzer.save_history(
            TrainingHistory(spec=spec, epochs=epochs, extras={}), str(path)
        )

    def test_plots_each_configuration_and_the_comparison(self, tmp_path: Path) -> None:
        """Per-run curves and a distribution go to the configuration's plot
        directory, and the cross-configuration comparison to the plots root. The
        configuration file in the history directory is not read as a history.
        Only the comparison gets the extra formats, with the PNG's name; the
        per-run plots stay PNG."""
        layout = ResultsLayout(str(tmp_path))
        specs = [
            build_spec("pl_qtl", seed=1),
            build_spec("pl_qtl", seed=2),
            build_spec("ctl", seed=1),
        ]
        for spec in specs:
            self._write_history(layout, spec)
            layout.write_config(spec)

        plotted = plot_losses(layout, vector_formats=("svg", "pdf"))

        qtl_plots = Path(layout.plots_dir("pl_qtl", config_id(specs[0])))
        for spec in specs[:2]:
            assert (qtl_plots / f"pl_qtl_{run_id(spec)}_loss.png").stat().st_size > 0
        assert (qtl_plots / "pl_qtl_loss_distribution.png").stat().st_size > 0
        assert not list(qtl_plots.glob("*.svg"))
        plots = Path(layout.root) / "plots"
        for extension in ("png", "svg", "pdf"):
            assert (plots / f"loss_comparison.{extension}").stat().st_size > 0
        assert len(plotted) == 2

    def test_unknown_vector_format_is_refused(self, tmp_path: Path) -> None:
        """A typo in a format fails before anything is drawn."""
        with pytest.raises(ValueError, match="Unknown vector format"):
            MetricsVisualizer(str(tmp_path), vector_formats=("eps",))

    def test_excluded_configuration_is_left_out_of_the_comparison(
        self, tmp_path: Path
    ) -> None:
        """An excluded configuration is missing from the comparison but keeps its
        own plots."""
        layout = ResultsLayout(str(tmp_path))
        kept, excluded = build_spec("pl_qtl", seed=1), build_spec("ctl", seed=1)
        for spec in (kept, excluded):
            self._write_history(layout, spec)
            layout.write_config(spec)

        plotted = plot_losses(layout, exclude=[config_id(excluded)])

        assert list(plotted.values()) == [layout.history_dir("pl_qtl", config_id(kept))]
        own_plots = Path(layout.plots_dir("ctl", config_id(excluded)))
        assert (own_plots / "ctl_loss_distribution.png").stat().st_size > 0

    def test_unknown_excluded_configuration_is_refused(self, tmp_path: Path) -> None:
        """A mistyped ID fails instead of drawing the figure with everything."""
        layout = ResultsLayout(str(tmp_path))
        spec = build_spec("pl_qtl", seed=1)
        self._write_history(layout, spec)
        layout.write_config(spec)

        with pytest.raises(ValueError, match="000000000000"):
            plot_losses(layout, exclude=["000000000000"])

    def test_no_histories_plots_nothing(self, tmp_path: Path) -> None:
        """An empty layout gives an empty result instead of blank figures."""
        assert plot_losses(ResultsLayout(str(tmp_path))) == {}


class TestPlotMetricComparisons:
    """Validates the cross-configuration metric comparison."""

    def test_excluded_configuration_is_left_out(self, tmp_path: Path) -> None:
        """Only the configurations not excluded are drawn, on every cohort."""
        layout = ResultsLayout(str(tmp_path))
        for paradigm in ("pl_qtl", "ctl"):
            _write_results(
                layout,
                build_spec(paradigm, seed=1),
                build_spec(paradigm, seed=2),
            )
        excluded = config_id(build_spec("ctl", seed=1))

        drawn = plot_metric_comparisons(layout, exclude=[excluded])

        assert set(drawn) == {"test", "val"}
        assert all(len(labels) == 1 for labels in drawn.values())
        assert all("pl_qtl" in labels[0] for labels in drawn.values())

    def test_unknown_excluded_configuration_is_refused(self, tmp_path: Path) -> None:
        """A mistyped ID fails instead of drawing the figure with everything."""
        layout = ResultsLayout(str(tmp_path))
        _write_results(
            layout, build_spec("pl_qtl", seed=1), build_spec("pl_qtl", seed=2)
        )

        with pytest.raises(ValueError, match="000000000000"):
            plot_metric_comparisons(layout, exclude=["000000000000"])
