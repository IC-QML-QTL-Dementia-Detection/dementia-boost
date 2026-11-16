"""Statistics over the runs of a configuration, beyond the mean and spread.

Each function reads run entries as the results files store them (the
`individual_runs` of a `{val,test}_results.json`), so the numbers are computed
from the same files as the report and regenerated with every evaluation.

- `majority_class_accuracy`: the accuracy of always answering the most frequent
  class of the cohort, the reference for "no skill".
- `count_collapsed_runs`: runs that predict a single class, or score at or
  below that reference.
"""

from collections.abc import Mapping, Sequence
from typing import Any

# The stored accuracy of an all-majority run and `majority_class_accuracy` are
# the same ratio computed two ways; they may differ in the last bit.
_ROUNDING: float = 1e-12


def majority_class_accuracy(runs: Sequence[Mapping[str, Any]]) -> float:
    """Computes the accuracy of always answering the most frequent class.

    Args:
        runs: Run entries with `y_true`, all scored on the same cohort.

    Returns:
        The share of the most frequent label.

    Raises:
        ValueError: If there are no runs, or the runs were scored on different
            labels (so they were not evaluated on the same slices).
    """
    if not runs:
        raise ValueError("No runs to read the labels from.")
    labels = list(runs[0]["y_true"])
    if any(list(run["y_true"]) != labels for run in runs):
        raise ValueError(
            "The runs were scored on different labels, so they were not "
            "evaluated on the same cohort."
        )
    positives = sum(labels) / len(labels)
    return max(positives, 1.0 - positives)


def count_collapsed_runs(
    runs: Sequence[Mapping[str, Any]], majority: float
) -> dict[str, int]:
    """Counts the runs that learned nothing usable.

    Read from each run's confusion matrix `[[TN, FP], [FN, TP]]` (scikit-learn's
    order, positive class 1 = Demented) at the 0.5 threshold.

    Args:
        runs: Run entries with `confusion_matrix` and `accuracy`.
        majority: The majority-class accuracy of the cohort.

    Returns:
        `runs` (how many), `only_negative` (every slice predicted Nondemented),
        `only_positive` (every slice predicted Demented), `recall_zero` (no
        Demented slice found, which includes `only_negative`), and
        `at_or_below_majority` (accuracy no better than always answering the
        most frequent class).
    """
    counts = {
        "runs": len(runs),
        "only_negative": 0,
        "only_positive": 0,
        "recall_zero": 0,
        "at_or_below_majority": 0,
    }
    for run in runs:
        (true_negative, false_positive), (false_negative, true_positive) = run[
            "confusion_matrix"
        ]
        if true_positive + false_positive == 0:
            counts["only_negative"] += 1
        if true_negative + false_negative == 0:
            counts["only_positive"] += 1
        if true_positive == 0 and false_negative > 0:
            counts["recall_zero"] += 1
        if run["accuracy"] <= majority + _ROUNDING:
            counts["at_or_below_majority"] += 1
    return counts
