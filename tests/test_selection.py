"""Unit tests for model selection on validation metrics.

Regression coverage
-------------------
- The best baseline backbone was chosen by test accuracy, and the same test
  cohort was then used to report every downstream result, which inflates them.
- The selection function was copied into three training scripts.
"""

import json
from pathlib import Path

import pytest
from conftest import build_spec

from dementia_boost.core.identity import config_id, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import MetricsAnalyzer, TrainingHistory
from dementia_boost.telemetry.selection import (
    select_backbone,
    select_best_baseline,
    select_best_run,
)


def _run(run_id: str, auc: float, f1: float = 0.5, loss: float = 0.7, **extra) -> dict:
    """Builds one per-run metrics entry as the evaluation scripts write it."""
    return {
        "run_id": run_id,
        "auc": auc,
        "f1_score": f1,
        "log_loss": loss,
        "accuracy": 0.5,
        **extra,
    }


class TestSelectBestRun:
    """Validates the ranking order and its tie-breaks."""

    def test_auc_is_the_primary_key(self) -> None:
        """A higher AUC wins even against better F1, accuracy, and loss."""
        runs = [
            _run("a", auc=0.80, f1=0.95, loss=0.1, accuracy=0.99),
            _run("b", auc=0.85, f1=0.10, loss=0.9),
        ]
        assert select_best_run(runs) == "b"

    def test_equal_auc_falls_back_to_f1(self) -> None:
        """With equal AUC the higher F1 wins."""
        runs = [_run("a", 0.8, f1=0.6), _run("b", 0.8, f1=0.7)]
        assert select_best_run(runs) == "b"

    def test_equal_auc_and_f1_fall_back_to_lowest_loss(self) -> None:
        """With equal AUC and F1 the lower validation loss wins."""
        runs = [_run("a", 0.8, 0.6, loss=0.5), _run("b", 0.8, 0.6, loss=0.4)]
        assert select_best_run(runs) == "b"

    def test_complete_tie_picks_the_smallest_run_id(self) -> None:
        """A full tie is resolved by run ID, so the choice never depends on order."""
        runs = [_run("seed_2", 0.8), _run("seed_1", 0.8)]
        assert select_best_run(runs) == "seed_1"
        assert select_best_run(list(reversed(runs))) == "seed_1"

    def test_test_metrics_are_ignored(self) -> None:
        """Fields from the test cohort must not change the choice."""
        runs = [
            _run("a", 0.80, test_auc=0.50, test_f1_score=0.0, test_accuracy=0.1),
            _run("b", 0.70, test_auc=1.00, test_f1_score=1.0, test_accuracy=1.0),
        ]
        assert select_best_run(runs) == "a"

    def test_run_without_a_validation_metric_raises(self) -> None:
        """A run lacking a ranking metric is rejected and named."""
        broken = {"run_id": "no_loss", "auc": 0.9, "f1_score": 0.8}
        with pytest.raises(ValueError, match="no_loss"):
            select_best_run([_run("a", 0.8), broken])

    def test_empty_input_raises(self) -> None:
        """Selecting from nothing is an error."""
        with pytest.raises(ValueError):
            select_best_run([])


class TestSelectBestBaseline:
    """Validates reading the validation results file."""

    def _write(self, path: Path, cohort: str | None, runs: list[dict]) -> str:
        """Writes a results file with the given cohort marker and runs."""
        payload: dict = {"individual_runs": runs}
        if cohort is not None:
            payload["cohort"] = cohort
        path.write_text(json.dumps(payload))
        return str(path)

    def test_reads_validation_results(self, tmp_path: Path) -> None:
        """A validation results file returns the best run ID."""
        path = self._write(
            tmp_path / "val.json", "val", [_run("baseline_seed_1", 0.7), _run("x", 0.9)]
        )
        assert select_best_baseline(path) == "x"

    @pytest.mark.parametrize("cohort", ["test", "train", None])
    def test_refuses_results_from_another_cohort(
        self, tmp_path: Path, cohort: str | None
    ) -> None:
        """Test metrics, or metrics of unknown origin, can never drive selection."""
        path = self._write(tmp_path / "r.json", cohort, [_run("a", 0.9)])
        with pytest.raises(ValueError, match="validation"):
            select_best_baseline(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing results file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            select_best_baseline(str(tmp_path / "absent.json"))


class TestSelectBackbone:
    """Validates resolving the selected baseline to its spec."""

    def _evaluated_baseline(
        self, layout: ResultsLayout, aucs: dict[int, float], **spec_overrides
    ) -> str:
        """Writes histories and a validation results file for baseline seeds.

        Args:
            layout: Layout to write into.
            aucs: Validation AUC per seed.
            **spec_overrides: Spec fields shared by the seeds (another
                configuration).

        Returns:
            The configuration ID of the written baseline.
        """
        specs = [build_spec("baseline", seed=s, **spec_overrides) for s in aucs]
        for spec in specs:
            path = Path(layout.history_path(spec))
            path.parent.mkdir(parents=True, exist_ok=True)
            MetricsAnalyzer.save_history(
                TrainingHistory(spec=spec, epochs=[], extras={}), str(path)
            )
        configuration = config_id(specs[0])
        results = Path(layout.metrics_path("baseline", configuration, "val"))
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text(
            json.dumps(
                {
                    "cohort": "val",
                    "individual_runs": [
                        _run(run_id(spec), auc=aucs[spec.seed]) for spec in specs
                    ],
                }
            )
        )
        return configuration

    def test_returns_the_spec_of_the_best_validation_run(self, tmp_path: Path) -> None:
        """The spec carries everything needed to find the checkpoint and to set
        the heads' `backbone_id`."""
        layout = ResultsLayout(str(tmp_path))
        self._evaluated_baseline(layout, {1: 0.6, 2: 0.9, 3: 0.7})

        spec = select_backbone(layout)

        assert spec == build_spec("baseline", seed=2)
        assert layout.checkpoint_path(spec).endswith("seed_2.pt")

    def test_an_explicit_configuration_is_used(self, tmp_path: Path) -> None:
        """With several evaluated baselines, naming one selects within it."""
        layout = ResultsLayout(str(tmp_path))
        first = self._evaluated_baseline(layout, {1: 0.6, 2: 0.9})
        self._evaluated_baseline(layout, {1: 0.99, 2: 0.5}, lr=5e-4)

        assert select_backbone(layout, first).seed == 2

    def test_several_baselines_without_a_choice_raise(self, tmp_path: Path) -> None:
        """Picking between configurations silently would be a hidden decision."""
        layout = ResultsLayout(str(tmp_path))
        self._evaluated_baseline(layout, {1: 0.6})
        self._evaluated_baseline(layout, {1: 0.7}, lr=5e-4)

        with pytest.raises(ValueError, match="configuration"):
            select_backbone(layout)

    def test_no_evaluated_baseline_says_what_to_run(self, tmp_path: Path) -> None:
        """Without validation results the error names the script to run."""
        with pytest.raises(FileNotFoundError, match="evaluate_baseline"):
            select_backbone(ResultsLayout(str(tmp_path)))

    def test_a_selected_run_without_a_history_raises(self, tmp_path: Path) -> None:
        """A results file naming a run that has no history is inconsistent."""
        layout = ResultsLayout(str(tmp_path))
        configuration = self._evaluated_baseline(layout, {1: 0.6, 2: 0.9})
        Path(layout.history_path(build_spec("baseline", seed=2))).unlink()

        with pytest.raises(ValueError, match="history"):
            select_backbone(layout, configuration)

    def test_test_results_cannot_drive_the_choice(self, tmp_path: Path) -> None:
        """A validation file that is really marked as test is refused."""
        layout = ResultsLayout(str(tmp_path))
        configuration = self._evaluated_baseline(layout, {1: 0.6, 2: 0.9})
        path = Path(layout.metrics_path("baseline", configuration, "val"))
        payload = json.loads(path.read_text())
        payload["cohort"] = "test"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="validation"):
            select_backbone(layout, configuration)
