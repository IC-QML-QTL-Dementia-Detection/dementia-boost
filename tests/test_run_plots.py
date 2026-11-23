"""Tests for the plotting steps on the results layout.

The tests check what the plotting steps ask the visualizer to draw (which runs,
which configurations, which formats), not the drawing: every drawing method is
replaced by a recorder, so no figure is built, rendered, or written. A drawing
mistake is fixed by plotting the saved results again; a wrong choice of run or
configuration would go unnoticed.

Regression coverage
-------------------
- The isolated ROC curve and confusion matrix were drawn for the run with the
  best test accuracy and AUC, so a run was chosen on the test cohort.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
_DRAWING_METHODS = (
    "plot_metric_distributions",
    "plot_comparative_roc",
    "plot_isolated_roc",
    "plot_confusion_matrix",
    "plot_loss_curve",
    "plot_loss_distribution",
    "plot_loss_comparison",
    "plot_metric_comparison",
)


@dataclass(frozen=True)
class _Call:
    """One drawing the visualizer was asked for.

    Attributes:
        method: Name of the `MetricsVisualizer` method.
        output_dir: Directory the figure would be written to.
        vector_formats: Extra formats the figure would also be written in.
        args: Positional arguments of the call.
        kwargs: Keyword arguments of the call.
    """

    method: str
    output_dir: Path
    vector_formats: tuple[str, ...]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@pytest.fixture(autouse=True)
def calls(monkeypatch: pytest.MonkeyPatch) -> list[_Call]:
    """Replaces every drawing method with a recorder of its calls.

    Returns:
        A list that fills with one entry per requested figure, in call order.
    """
    records: list[_Call] = []

    def recorder(method: str):
        def record(self: MetricsVisualizer, *args: Any, **kwargs: Any) -> None:
            records.append(
                _Call(
                    method,
                    Path(self.output_dir),
                    self.vector_formats,
                    args,
                    kwargs,
                )
            )

        return record

    for method in _DRAWING_METHODS:
        monkeypatch.setattr(MetricsVisualizer, method, recorder(method))
    return records


def _of(calls: list[_Call], method: str) -> list[_Call]:
    """Returns the recorded calls of one drawing method."""
    return [call for call in calls if call.method == method]


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
    """Validates the plots requested for each configuration of a paradigm."""

    def test_isolated_plots_show_the_run_selected_on_validation(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """Every plot of the configuration is drawn from its test results into
        its plot directory, with the requested formats. The isolated plots show
        run B, the best on validation, not run A, the best on test."""
        layout = ResultsLayout(str(tmp_path))
        spec_a = build_spec("qiskit_qtl", seed=1)
        spec_b = build_spec("qiskit_qtl", seed=2)
        _write_results(layout, spec_a, spec_b)

        plotted = plot_paradigm(
            layout, Paradigm.QISKIT_QTL, vector_formats=("svg", "pdf")
        )

        config = config_id(spec_a)
        assert plotted == [config]
        assert len(calls) == 4
        test_path = layout.metrics_path("qiskit_qtl", config, "test")
        assert all(call.args == (test_path,) for call in calls)
        assert all(
            call.output_dir == Path(layout.plots_dir("qiskit_qtl", config))
            and call.vector_formats == ("svg", "pdf")
            for call in calls
        )
        for method in ("plot_isolated_roc", "plot_confusion_matrix"):
            (call,) = _of(calls, method)
            assert call.kwargs["run_id"] == run_id(spec_b)

    def test_nothing_to_plot_returns_nothing(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """A paradigm without results requests no plots."""
        assert plot_paradigm(ResultsLayout(str(tmp_path)), Paradigm.CTL) == []
        assert calls == []


class TestPlotLosses:
    """Validates the loss plots requested from the histories on disk."""

    def _write_history(self, layout: ResultsLayout, spec: RunSpec) -> None:
        """Writes a two-epoch history for a spec, and its configuration file."""
        path = Path(layout.history_path(spec))
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs = [
            EpochRecord(
                epoch=e,
                train_loss=0.7,
                train_acc=0.5,
                val_loss=0.7,
                val_acc=0.5,
                lr=spec.lr,
                duration_s=1.0,
            )
            for e in (1, 2)
        ]
        MetricsAnalyzer.save_history(
            TrainingHistory(spec=spec, epochs=epochs, extras={}), str(path)
        )
        layout.write_config(spec)

    def test_plots_each_configuration_and_the_comparison(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """Each run gets a loss curve and each configuration a distribution, in
        its plot directory and without extra formats; the comparison covers
        every configuration and alone gets the extra formats."""
        layout = ResultsLayout(str(tmp_path))
        specs = [
            build_spec("qiskit_qtl", seed=1),
            build_spec("qiskit_qtl", seed=2),
            build_spec("ctl", seed=1),
        ]
        for spec in specs:
            self._write_history(layout, spec)

        plotted = plot_losses(layout, vector_formats=("svg", "pdf"))

        curves = _of(calls, "plot_loss_curve")
        assert {call.args[0] for call in curves} == {
            layout.history_path(spec) for spec in specs
        }
        distributions = _of(calls, "plot_loss_distribution")
        assert {call.args[0] for call in distributions} == {
            layout.history_dir(spec.paradigm, config_id(spec)) for spec in specs
        }
        assert all(call.vector_formats == () for call in curves + distributions)
        (comparison,) = _of(calls, "plot_loss_comparison")
        assert comparison.vector_formats == ("svg", "pdf")
        assert sorted(comparison.args[0].values()) == sorted(plotted.values())
        assert len(plotted) == 2

    def test_unknown_vector_format_is_refused(self, tmp_path: Path) -> None:
        """A typo in a format fails before anything is drawn."""
        with pytest.raises(ValueError, match="Unknown vector format"):
            MetricsVisualizer(str(tmp_path), vector_formats=("eps",))

    def test_excluded_configuration_is_left_out_of_the_comparison(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """An excluded configuration is missing from the comparison but keeps its
        own plots."""
        layout = ResultsLayout(str(tmp_path))
        kept, excluded = build_spec("qiskit_qtl", seed=1), build_spec("ctl", seed=1)
        for spec in (kept, excluded):
            self._write_history(layout, spec)

        plot_losses(layout, exclude=[config_id(excluded)])

        (comparison,) = _of(calls, "plot_loss_comparison")
        assert list(comparison.args[0].values()) == [
            layout.history_dir("qiskit_qtl", config_id(kept))
        ]
        excluded_dir = layout.history_dir("ctl", config_id(excluded))
        assert excluded_dir in {
            call.args[0] for call in _of(calls, "plot_loss_distribution")
        }

    def test_unknown_excluded_configuration_is_refused(self, tmp_path: Path) -> None:
        """A mistyped ID fails instead of drawing the figure with everything."""
        layout = ResultsLayout(str(tmp_path))
        self._write_history(layout, build_spec("qiskit_qtl", seed=1))

        with pytest.raises(ValueError, match="000000000000"):
            plot_losses(layout, exclude=["000000000000"])

    def test_no_histories_plots_nothing(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """An empty layout requests no figure."""
        assert plot_losses(ResultsLayout(str(tmp_path))) == {}
        assert calls == []


class TestPlotMetricComparisons:
    """Validates the cross-configuration metric comparison."""

    def test_excluded_configuration_is_left_out(
        self, tmp_path: Path, calls: list[_Call]
    ) -> None:
        """Only the configurations not excluded are drawn, on every cohort."""
        layout = ResultsLayout(str(tmp_path))
        for paradigm in ("qiskit_qtl", "ctl"):
            _write_results(
                layout,
                build_spec(paradigm, seed=1),
                build_spec(paradigm, seed=2),
            )
        excluded = config_id(build_spec("ctl", seed=1))

        drawn = plot_metric_comparisons(layout, exclude=[excluded])

        assert set(drawn) == {"test", "val"}
        assert all(len(labels) == 1 for labels in drawn.values())
        assert all("qiskit_qtl" in labels[0] for labels in drawn.values())
        assert [len(call.args[0]) for call in calls] == [1, 1]

    def test_unknown_excluded_configuration_is_refused(self, tmp_path: Path) -> None:
        """A mistyped ID fails instead of drawing the figure with everything."""
        layout = ResultsLayout(str(tmp_path))
        _write_results(
            layout, build_spec("qiskit_qtl", seed=1), build_spec("qiskit_qtl", seed=2)
        )

        with pytest.raises(ValueError, match="000000000000"):
            plot_metric_comparisons(layout, exclude=["000000000000"])
