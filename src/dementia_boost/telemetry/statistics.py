"""Statistics over the runs of a configuration, beyond the mean and spread.

Each function reads run entries as the results files store them (the
`individual_runs` of a `{val,test}_results.json`), so the numbers are computed
from the same files as the report and regenerated with every evaluation.

- `majority_class_accuracy`: the accuracy of always answering the most frequent
  class of the cohort, the reference for "no skill".
- `count_collapsed_runs`: runs that predict a single class, or score at or
  below that reference.
- `compare_paired`: two configurations compared seed by seed (Wilcoxon
  signed-rank test, Hodges-Lehmann estimate with an interval, Holm-adjusted
  p-values over the metrics). Seeds are recovered from the run IDs with
  `seeds_of_runs`; `share_initial_weights` says whether one seed gave both
  configurations the same initial weights, which is what makes the pairing
  informative.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy.stats import norm, wilcoxon

from dementia_boost.core.identity import RunSpec, run_id

SEED_SEARCH_LIMIT: int = 10_000
CONFIDENCE: float = 0.95

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


def seeds_of_runs(payload: Mapping[str, Any]) -> dict[str, int]:
    """Recovers the seed of every run in a results payload.

    A `run_id` hashes the configuration's spec plus the seed, so hashing the
    stored spec with candidate seeds finds each run's seed; nothing is parsed.

    Args:
        payload: A results payload with `configuration` and `individual_runs`.

    Returns:
        Mapping of run ID to its seed.

    Raises:
        ValueError: If a run's ID matches no seed below `SEED_SEARCH_LIMIT`, so
            the run does not belong to the payload's configuration.
    """
    spec = payload["configuration"]["spec"]
    wanted = {run["run_id"] for run in payload["individual_runs"]}
    seeds: dict[str, int] = {}
    for seed in range(SEED_SEARCH_LIMIT):
        identifier = run_id(RunSpec.from_dict({**spec, "seed": seed}))
        if identifier in wanted:
            seeds[identifier] = seed
            if len(seeds) == len(wanted):
                return seeds
    missing = sorted(wanted - set(seeds))
    raise ValueError(
        f"Runs {missing} match no seed of configuration "
        f"{payload['configuration']['config_id']!r}."
    )


def share_initial_weights(first: RunSpec, second: RunSpec) -> bool:
    """Says whether one seed gives two configurations the same initial weights.

    The weights are drawn in a fixed order, so the same seed gives the same
    draws when the heads have the same parameters: same paradigm, qubits,
    layers, and variational block. The embedding and the learning rates do not
    change the shapes.

    Args:
        first: Spec of a run of the first configuration.
        second: Spec of a run of the second configuration.

    Returns:
        True if seed `s` starts both configurations from the same weights.
    """
    fields = ("paradigm", "n_qubits", "n_layers", "block")
    return all(getattr(first, name) == getattr(second, name) for name in fields)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Adjusts p-values for several comparisons with Holm's step-down method.

    Args:
        p_values: Mapping of test name to its p-value.

    Returns:
        Mapping of test name to its adjusted p-value, never below the raw one
        and at most 1.
    """
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _hodges_lehmann(differences: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Estimates the centre of paired differences, with a confidence interval.

    The estimate is the median of the Walsh averages `(d_i + d_j) / 2` for
    `i <= j`; the interval takes the order statistics of the Walsh averages
    that match the Wilcoxon signed-rank test, with the normal approximation of
    its critical value.

    Args:
        differences: The paired differences.

    Returns:
        The estimate and the `CONFIDENCE` interval.
    """
    n = len(differences)
    i, j = np.triu_indices(n)
    walsh = np.sort((differences[i] + differences[j]) / 2.0)
    total = len(walsh)
    spread = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    critical = norm.ppf(0.5 + CONFIDENCE / 2.0)
    lower = max(0, math.floor(n * (n + 1) / 4.0 - critical * spread))
    return float(np.median(walsh)), (
        float(walsh[lower]),
        float(walsh[total - 1 - lower]),
    )


def compare_paired(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    metrics: Sequence[str],
) -> dict[str, Any]:
    """Compares two configurations seed by seed on one cohort.

    For every seed both configurations ran, the difference `first - second`
    of each metric is formed. Per metric the result gives the number of seeds
    where the first configuration is higher (wins), lower (losses), or equal
    (ties), the median difference, the Hodges-Lehmann estimate with its
    interval, and the Wilcoxon signed-rank p-value (ties dropped), raw and
    Holm-adjusted over the metrics.

    Args:
        first: Results payload of the first configuration.
        second: Results payload of the second configuration.
        metrics: Names of the metrics to compare.

    Returns:
        `cohort`, `pairs` (shared seeds), `only_first` and `only_second` (seeds
        on one side only), `same_initial_weights` (see `share_initial_weights`),
        and `metrics` (per metric, the statistics above).

    Raises:
        ValueError: If the payloads are of different cohorts or splits, or
            share no seed.
    """
    if first["cohort"] != second["cohort"]:
        raise ValueError(
            f"Cannot pair a {first['cohort']!r} with a {second['cohort']!r} result."
        )
    first_spec = first["configuration"]["spec"]
    second_spec = second["configuration"]["spec"]
    if first_spec["split_id"] != second_spec["split_id"]:
        raise ValueError(
            "The configurations were trained on different splits: "
            f"{first_spec['split_id']} and {second_spec['split_id']}."
        )

    first_runs = _runs_by_seed(first)
    second_runs = _runs_by_seed(second)
    shared = sorted(set(first_runs) & set(second_runs))
    if not shared:
        raise ValueError("The two configurations share no seed to pair.")

    compared: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        differences = np.array(
            [first_runs[s][metric] - second_runs[s][metric] for s in shared],
            dtype=float,
        )
        estimate, interval = _hodges_lehmann(differences)
        nonzero = differences[differences != 0.0]
        if len(nonzero) == 0:
            statistic, p_value = 0.0, 1.0
        else:
            raw_statistic, raw_p_value = cast(tuple[float, float], wilcoxon(nonzero))
            statistic, p_value = float(raw_statistic), float(raw_p_value)
        compared[metric] = {
            "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
            "median_difference": float(np.median(differences)),
            "hodges_lehmann": estimate,
            "interval": interval,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        }
    adjusted = holm_adjust({name: compared[name]["p_value"] for name in compared})
    for name, value in adjusted.items():
        compared[name]["p_value_holm"] = value

    return {
        "cohort": first["cohort"],
        "pairs": len(shared),
        "only_first": len(first_runs) - len(shared),
        "only_second": len(second_runs) - len(shared),
        "same_initial_weights": share_initial_weights(
            RunSpec.from_dict({**first_spec, "seed": 0}),
            RunSpec.from_dict({**second_spec, "seed": 0}),
        ),
        "metrics": compared,
    }


def _runs_by_seed(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """Indexes the runs of a results payload by their seed.

    Args:
        payload: A results payload.

    Returns:
        Mapping of seed to the run's entry.
    """
    seeds = seeds_of_runs(payload)
    return {seeds[run["run_id"]]: run for run in payload["individual_runs"]}
