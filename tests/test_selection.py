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

from dementia_boost.telemetry.selection import select_best_baseline, select_best_run


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
