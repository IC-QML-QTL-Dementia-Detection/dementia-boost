"""Unit tests for the statistics computed over the runs of a configuration.

Regression coverage
-------------------
- The counts of collapsed runs and the majority-class accuracy cited in the
  results were computed by hand, outside the code that produces the results.
"""

import pytest

from dementia_boost.telemetry.statistics import (
    count_collapsed_runs,
    majority_class_accuracy,
)

_LABELS = [0, 0, 0, 1, 1]


def _run(confusion: list[list[int]], accuracy: float, labels=_LABELS) -> dict:
    """Builds one run entry as the results files store it."""
    return {"confusion_matrix": confusion, "accuracy": accuracy, "y_true": labels}


class TestMajorityClassAccuracy:
    """Validates the accuracy of always answering the most frequent class."""

    def test_is_the_share_of_the_most_frequent_label(self) -> None:
        """Three of five labels are 0, so always answering 0 scores 0.6."""
        assert majority_class_accuracy([_run([[3, 0], [2, 0]], 0.6)]) == 0.6

    def test_runs_scored_on_different_labels_are_refused(self) -> None:
        """Every run of a cohort must be scored on the same slices."""
        runs = [_run([[3, 0], [2, 0]], 0.6), _run([[3, 0], [2, 0]], 0.6, [0, 1])]
        with pytest.raises(ValueError, match="labels"):
            majority_class_accuracy(runs)


class TestCountCollapsedRuns:
    """Validates the counts of runs that predict a single class."""

    def test_counts_each_kind_of_failure(self) -> None:
        """An all-negative run, an all-positive run, and a mixed run with no
        true positive are told apart; runs at or below the majority-class
        accuracy are counted separately."""
        runs = [
            _run([[3, 0], [2, 0]], 0.6),
            _run([[0, 3], [0, 2]], 0.4),
            _run([[2, 1], [2, 0]], 0.4),
            _run([[3, 0], [0, 2]], 1.0),
        ]

        counts = count_collapsed_runs(runs, majority=0.6)

        assert counts == {
            "runs": 4,
            "only_negative": 1,
            "only_positive": 1,
            "recall_zero": 2,
            "at_or_below_majority": 3,
        }
