"""Unit tests for the stateless metrics engine, logger, and visualizer.

This module validates:
- ``MetricsAnalyzer.calculate_metrics`` exact classification metric values,
  zero-division safety, and threshold sensitivity.
- ``MetricsAnalyzer.aggregate_results`` cross-run statistical aggregation
  and empty-input safety.
- ``MetricsAnalyzer.save_to_json`` schema integrity through a round trip.
- ``MetricsAnalyzer.save_history`` / ``load_history`` lossless round trip of
  ``TrainingHistory``, including epochs without validation values.
- ``setup_logger`` dual-output configuration and singleton handler reuse.
- ``MetricsVisualizer`` PNG artifact generation across all plotting methods.
- ``MetricsVisualizer`` loss-curve plots tolerating sparse validation values,
  ragged epoch counts, and empty history directories.

Regression coverage
-------------------
- Silent drift in metric formulas producing incorrect Accuracy, Precision,
  Recall, F1, or AUC values.
- Division-by-zero exceptions from degenerate all-negative predictions.
- Threshold changes failing to propagate into the computed confusion matrix.
- Aggregate statistics diverging from ground-truth NumPy computations.
- Corrupted or incomplete JSON telemetry payloads breaking downstream
  visualization scripts.
- Per-epoch training histories losing fields or ``None`` validation values
  when persisted and reloaded.
- Duplicate log handlers accumulating across repeated `setup_logger` calls.
- Visualization methods silently failing to write plot artifacts to disk.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from conftest import build_spec

from dementia_boost.core.identity import run_id
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    EvaluationResult,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.viz.visualizer import MetricsVisualizer

_KNOWN_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc",
)


class TestMetricsAnalyzerCalculateMetricsValues:
    """Validates exact metric outputs against known ground truth."""

    def test_known_predictions_yield_exact_metric_values(self) -> None:
        """Asserts exact Accuracy, Precision, Recall, F1, AUC, and confusion
        matrix values for a fixed synthetic prediction/ground-truth pair."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])

        result = MetricsAnalyzer.calculate_metrics("known_case", y_true, y_prob)

        assert result.accuracy == pytest.approx(0.75)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(0.5)
        assert result.f1_score == pytest.approx(2 / 3)
        assert result.auc == pytest.approx(0.75)
        assert result.confusion_matrix == [[2, 0], [1, 1]]
        expected_log_loss = -np.mean(np.log([0.9, 0.6, 0.35, 0.8]))
        assert result.log_loss == pytest.approx(expected_log_loss)

    def test_log_loss_stays_finite_for_saturated_probabilities(self) -> None:
        """A wrong prediction at exactly 0 or 1 must not give an infinite loss."""
        result = MetricsAnalyzer.calculate_metrics(
            "saturated", np.array([1, 0]), np.array([0.0, 1.0])
        )
        assert np.isfinite(result.log_loss)


class TestMetricsAnalyzerZeroDivisionSafety:
    """Validates degenerate prediction handling without exceptions."""

    def test_no_positive_predictions_guards_precision_and_zeroes_recall(
        self,
    ) -> None:
        """When every sample is predicted negative, precision's zero-division
        (TP + FP == 0) must be safely guarded to 0.0, and recall naturally
        evaluates to 0.0 since no true positives were captured."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.05, 0.1, 0.2, 0.3])

        result = MetricsAnalyzer.calculate_metrics("all_negative", y_true, y_prob)

        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_all_positive_predictions_do_not_raise_and_stay_bounded(self) -> None:
        """When every sample is predicted positive, no exception should be
        raised and precision/recall must remain within [0.0, 1.0]."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])

        result = MetricsAnalyzer.calculate_metrics("all_positive", y_true, y_prob)

        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0


class TestMetricsAnalyzerCustomThreshold:
    """Validates that shifting the decision threshold alters the outcome."""

    def test_threshold_shift_changes_confusion_matrix_and_metrics(self) -> None:
        """Asserts that a lenient threshold (0.3) versus a strict threshold
        (0.7) produce distinct confusion matrices and metric values that
        match the expected shift in the underlying decision boundary."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.4, 0.6, 0.8])

        lenient = MetricsAnalyzer.calculate_metrics(
            "lenient",
            y_true,
            y_prob,
            threshold=0.3,
        )
        strict = MetricsAnalyzer.calculate_metrics(
            "strict",
            y_true,
            y_prob,
            threshold=0.7,
        )

        assert lenient.confusion_matrix != strict.confusion_matrix
        assert lenient.confusion_matrix == [[1, 1], [0, 2]]
        assert strict.confusion_matrix == [[2, 0], [1, 1]]
        assert lenient.recall == pytest.approx(1.0)
        assert strict.recall == pytest.approx(0.5)
        assert lenient.precision == pytest.approx(2 / 3)
        assert strict.precision == pytest.approx(1.0)


class TestMetricsAnalyzerAggregation:
    """Validates cross-run statistical aggregation."""

    def _make_result(self, run_id: str, value: float) -> EvaluationResult:
        """Builds an `EvaluationResult` with every metric set to `value`.

        Args:
            run_id: Unique identifier for the synthetic run.
            value: Scalar value assigned uniformly to every metric field.

        Returns:
            A populated `EvaluationResult` DTO.
        """
        return EvaluationResult(
            run_id=run_id,
            accuracy=value,
            precision=value,
            recall=value,
            f1_score=value,
            auc=value,
            log_loss=value,
            confusion_matrix=[[1, 0], [0, 1]],
            y_true=[0, 1],
            y_prob=[0.1, 0.9],
        )

    def test_aggregation_matches_numpy_statistics(self) -> None:
        """Asserts sample mean, standard deviation, minimum, and maximum
        match direct NumPy computation across multiple synthetic runs."""
        values = [0.7, 0.8, 0.9, 0.6]
        results = [self._make_result(f"run_{i}", v) for i, v in enumerate(values)]

        aggregated = MetricsAnalyzer.aggregate_results(results)

        for key in _KNOWN_METRIC_KEYS:
            summary = aggregated[key]
            assert summary.mean == pytest.approx(float(np.mean(values)))
            assert summary.std == pytest.approx(float(np.std(values)))
            assert summary.min_val == pytest.approx(float(np.min(values)))
            assert summary.max_val == pytest.approx(float(np.max(values)))

    def test_empty_results_returns_empty_dict(self) -> None:
        """Asserts that aggregating an empty list of results is a safe no-op
        returning an empty dictionary rather than raising."""
        assert MetricsAnalyzer.aggregate_results([]) == {}


class TestMetricsAnalyzerSaveToJson:
    """Validates JSON serialization schema integrity."""

    def test_json_round_trip_preserves_schema(self, tmp_path: Path) -> None:
        """Serializes results to a temporary file, re-reads it, and asserts
        full JSON schema integrity across both top-level sections."""
        result = MetricsAnalyzer.calculate_metrics(
            "run_json",
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
        )
        aggregated = MetricsAnalyzer.aggregate_results([result])
        filepath = tmp_path / "metrics.json"

        configuration = {"config_id": "abc123", "label": "qtl", "spec": {"lr": 1e-4}}

        MetricsAnalyzer.save_to_json(
            [result],
            aggregated,
            str(filepath),
            cohort="val",
            configuration=configuration,
        )

        with open(filepath) as f:
            payload = json.load(f)

        assert set(payload.keys()) == {
            "cohort",
            "configuration",
            "aggregated_statistics",
            "individual_runs",
        }
        assert payload["cohort"] == "val"
        assert payload["configuration"] == configuration
        assert payload["individual_runs"][0]["run_id"] == "run_json"

        for key in _KNOWN_METRIC_KEYS:
            assert key in payload["aggregated_statistics"]
            assert payload["aggregated_statistics"][key]["mean"] == pytest.approx(
                getattr(result, key)
            )


class TestTrainingHistorySerialization:
    """Validates lossless JSON persistence of per-epoch training histories."""

    def test_history_round_trip_is_lossless(self, tmp_path: Path) -> None:
        """Saves a history mixing evaluated and skipped epochs, reloads it, and
        asserts equality, so ``None`` validation values, the spec, and the
        extras survive the trip."""
        history = self._history()
        filepath = tmp_path / "history.json"

        MetricsAnalyzer.save_history(history, str(filepath))

        assert MetricsAnalyzer.load_history(str(filepath)) == history

    def _history(self) -> TrainingHistory:
        """A two-epoch PL QTL history with a spec and extras."""
        return TrainingHistory(
            spec=build_spec(
                "qtl", seed=7, lr=1e-4, batch_size=64, n_qubits=6, n_layers=4
            ),
            extras={"quantum_device": "lightning.qubit"},
            epochs=[
                EpochRecord(1, 0.69, 0.51, 0.70, 0.50, 1e-4, 1.5),
                EpochRecord(2, 0.65, 0.58, None, None, 1e-4, 1.4),
            ],
        )

    def test_saved_file_stores_the_run_id_for_humans(self, tmp_path: Path) -> None:
        """The JSON carries the run ID next to the spec, so a hash can be found
        by searching the files."""
        history = self._history()
        filepath = tmp_path / "history.json"

        MetricsAnalyzer.save_history(history, str(filepath))

        payload = json.loads(filepath.read_text())
        assert payload["run_id"] == run_id(history.spec)
        assert payload["spec"]["n_qubits"] == 6

    def test_loading_a_history_whose_run_id_does_not_match_its_spec_raises(
        self, tmp_path: Path
    ) -> None:
        """A hand-edited spec, or a history written under another identity
        scheme, is refused instead of being trusted."""
        history = self._history()
        filepath = tmp_path / "history.json"
        MetricsAnalyzer.save_history(history, str(filepath))
        payload = json.loads(filepath.read_text())
        payload["spec"]["n_layers"] = 3
        filepath.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="run_id"):
            MetricsAnalyzer.load_history(str(filepath))


class TestSetupLogger:
    """Validates dual-output configuration and singleton reuse."""

    def test_dual_handlers_configured_and_reused_across_calls(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that a fresh logger receives exactly one console handler
        and one file handler, and that a repeated call with the same name
        returns the identical instance without adding duplicate handlers."""
        log_dir = str(tmp_path / "logs")
        logger_name = "test_logger_singleton"

        first_logger = setup_logger(logger_name, log_dir=log_dir)
        console_handlers = [
            h for h in first_logger.handlers if not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [
            h for h in first_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 1
        assert len(file_handlers) == 1

        second_logger = setup_logger(logger_name, log_dir=log_dir)

        assert second_logger is first_logger
        assert len(second_logger.handlers) == len(first_logger.handlers)


class TestMetricsVisualizerPlotGeneration:
    """Validates that all plotting methods emit non-empty PNG artifacts."""

    def _write_mock_metrics_json(self, tmp_path: Path) -> str:
        """Builds a small multi-run evaluation JSON file for plotting tests.

        Args:
            tmp_path: pytest-provided temporary directory.

        Returns:
            Absolute path string of the written JSON telemetry file.
        """
        results = [
            MetricsAnalyzer.calculate_metrics(
                f"seed_{i}",
                np.array([0, 0, 1, 1]),
                np.array([0.1 + 0.05 * i, 0.3, 0.6, 0.9 - 0.05 * i]),
            )
            for i in range(3)
        ]
        aggregated = MetricsAnalyzer.aggregate_results(results)
        filepath = tmp_path / "mock_metrics.json"
        MetricsAnalyzer.save_to_json(
            results, aggregated, str(filepath), cohort="test", configuration={}
        )
        return str(filepath)

    def test_all_plot_methods_write_nonzero_png_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Feeds mock evaluation JSON into every plotting method and asserts
        that each expected PNG file is created on disk with non-zero size."""
        json_path = self._write_mock_metrics_json(tmp_path)
        output_dir = str(tmp_path / "plots")
        visualizer = MetricsVisualizer(output_dir=output_dir)

        visualizer.plot_metric_distributions(json_path, prefix="test")
        visualizer.plot_comparative_roc(json_path, prefix="test")
        visualizer.plot_isolated_roc(json_path, run_id="seed_0", prefix="test")
        visualizer.plot_confusion_matrix(json_path, run_id="seed_0", prefix="test")

        expected_filenames = (
            "test_distributions.png",
            "test_roc_curves.png",
            "test_isolated_roc_seed_0.png",
            "test_cm_seed_0.png",
        )
        for filename in expected_filenames:
            artifact_path = Path(output_dir) / filename
            assert artifact_path.exists()
            assert artifact_path.stat().st_size > 0


class TestMetricsVisualizerLossPlots:
    """Validates the loss-curve plots against histories with awkward shapes."""

    def _write_histories(self, directory: Path, n_runs: int = 3) -> Path:
        """Writes histories with sparse validation values and unequal lengths.

        The first run is shorter than the others and every run only evaluates
        on even epochs, so the plots must tolerate missing values and ragged
        epoch counts.

        Args:
            directory: Directory receiving one JSON file per run.
            n_runs: Number of histories to write.

        Returns:
            The directory containing the written histories.
        """
        directory.mkdir(parents=True)
        for i in range(n_runs):
            n_epochs = 4 if i == 0 else 6
            epochs = [
                EpochRecord(
                    epoch=e,
                    train_loss=0.7 - 0.05 * e + 0.01 * i,
                    train_acc=0.5 + 0.05 * e,
                    val_loss=0.72 - 0.04 * e if e % 2 == 0 else None,
                    val_acc=0.5 + 0.04 * e if e % 2 == 0 else None,
                    lr=1e-4,
                    duration_s=1.0,
                )
                for e in range(1, n_epochs + 1)
            ]
            history = TrainingHistory(
                spec=build_spec("qtl", seed=i, lr_step_size=2),
                extras={},
                epochs=epochs,
            )
            MetricsAnalyzer.save_history(history, str(directory / f"seed_{i}.json"))
        return directory

    def test_loss_plots_write_nonzero_png_files(self, tmp_path: Path) -> None:
        """Feeds sparse, ragged histories into all three loss plots and asserts
        each expected PNG exists with non-zero size."""
        qtl_dir = self._write_histories(tmp_path / "qtl")
        ctl_dir = self._write_histories(tmp_path / "ctl")
        output_dir = Path(tmp_path / "plots")
        visualizer = MetricsVisualizer(output_dir=str(output_dir))

        visualizer.plot_loss_curve(str(qtl_dir / "seed_1.json"), prefix="qtl")
        visualizer.plot_loss_distribution(str(qtl_dir), prefix="qtl")
        visualizer.plot_loss_comparison({"QTL": str(qtl_dir), "CTL": str(ctl_dir)})

        seed_1_id = run_id(build_spec("qtl", seed=1, lr_step_size=2))
        for filename in (
            f"qtl_{seed_1_id}_loss.png",
            "qtl_loss_distribution.png",
            "loss_comparison.png",
        ):
            assert (output_dir / filename).stat().st_size > 0

    def test_loss_distribution_rejects_directory_without_histories(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that an empty history directory raises instead of silently
        writing a blank figure."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        visualizer = MetricsVisualizer(output_dir=str(tmp_path / "plots"))

        with pytest.raises(ValueError, match="No training histories"):
            visualizer.plot_loss_distribution(str(empty_dir), prefix="qtl")
