"""Statistics over the runs of a configuration, beyond the mean and spread.

Each function reads run entries as the results files store them (the
`individual_runs` of a `{val,test}_results.json`), so the numbers are computed
from the same files as the report and regenerated with every evaluation.

- `majority_class_accuracy`: the accuracy of always answering the most frequent
  class of the cohort, the reference for "no skill".
- `count_collapsed_runs`: runs that predict a single class, or score at or
  below that reference, and runs whose output is the same for every slice.
- `compare_paired`: two configurations compared seed by seed (Wilcoxon
  signed-rank test, Hodges-Lehmann estimate with an interval, Holm-adjusted
  p-values over the metrics). Seeds are recovered from the run IDs with
  `seeds_of_runs`; `share_initial_weights` says whether one seed gave both
  configurations the same initial weights, which is what makes the pairing
  informative.
- `bootstrap_patients`: metrics per patient (the mean probability of each
  patient's slices) and a class-stratified bootstrap over the patients, for one
  configuration or the paired difference of two. Unlike the seed spread, it
  measures how the numbers would move with other patients of the same kind.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy.stats import norm, rankdata, wilcoxon

from dementia_boost.core.identity import RunSpec, run_id

SEED_SEARCH_LIMIT: int = 10_000
CONFIDENCE: float = 0.95
DEFAULT_RESAMPLES: int = 2_000
THRESHOLD: float = 0.5
# Largest spread of a run's probabilities that still counts as one output. Far
# below any difference a working head makes between slices, and above the
# float32 noise of a head that outputs the same logit for every slice.
CONSTANT_TOLERANCE: float = 1e-6

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
    order, positive class 1 = Demented) at the 0.5 threshold, and from its
    probabilities. A run predicts a single class in one of two ways: its
    outputs vary but all fall on one side of the threshold, or its output is
    the same for every slice (for example a head whose hidden units are all
    inactive); `constant_output` counts the second kind.

    Args:
        runs: Run entries with `confusion_matrix`, `accuracy`, and `y_prob`.
        majority: The majority-class accuracy of the cohort.

    Returns:
        `runs` (how many), `only_negative` (every slice predicted Nondemented),
        `only_positive` (every slice predicted Demented), `recall_zero` (no
        Demented slice found, which includes `only_negative`),
        `at_or_below_majority` (accuracy no better than always answering the
        most frequent class), and `constant_output` (probabilities spread by
        at most `CONSTANT_TOLERANCE`, which implies a single class).
    """
    counts = {
        "runs": len(runs),
        "only_negative": 0,
        "only_positive": 0,
        "recall_zero": 0,
        "at_or_below_majority": 0,
        "constant_output": 0,
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
        if np.ptp(run["y_prob"]) <= CONSTANT_TOLERANCE:
            counts["constant_output"] += 1
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
    layers, and variational block, or the same classical head and width. The
    embedding and the learning rates do not change the shapes.

    Args:
        first: Spec of a run of the first configuration.
        second: Spec of a run of the second configuration.

    Returns:
        True if seed `s` starts both configurations from the same weights.
    """
    fields = ("paradigm", "n_qubits", "n_layers", "block", "head", "hidden_features")
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


def patient_probabilities(
    payload: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Turns the slice predictions of every run into one prediction per patient.

    A patient's probability is the mean of the probabilities of its slices; its
    label is the label of its slices, which is constant per subject.

    Args:
        payload: A results payload with `subject_ids` and `individual_runs`.

    Returns:
        The patients (sorted Subject IDs), their labels of shape `(n_patients,)`,
        and the probabilities of shape `(n_runs, n_patients)`.

    Raises:
        ValueError: If the payload has no subject IDs (it was evaluated before
            they were recorded), the runs were scored on different labels, or a
            patient's slices carry two labels.
    """
    subjects = payload.get("subject_ids")
    if subjects is None:
        raise ValueError(
            "These results have no subject IDs; re-run the evaluation script so "
            "they are recorded."
        )
    runs = payload["individual_runs"]
    labels = np.asarray(runs[0]["y_true"])
    if any(not np.array_equal(run["y_true"], labels) for run in runs):
        raise ValueError("The runs were scored on different labels.")

    patients = sorted(set(subjects))
    column = np.array([patients.index(subject) for subject in subjects])
    membership = np.eye(len(patients))[column]
    patient_labels = np.empty(len(patients), dtype=int)
    for index, patient in enumerate(patients):
        found = set(labels[column == index].tolist())
        if len(found) != 1:
            raise ValueError(f"Patient {patient} has slices with labels {found}.")
        patient_labels[index] = found.pop()

    probabilities = np.array([run["y_prob"] for run in runs], dtype=float)
    means = probabilities @ membership / membership.sum(axis=0)
    return patients, patient_labels, means


def stratified_resamples(labels: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Draws patients with replacement, separately within each class.

    Every resample keeps the class counts of the cohort, so both classes are
    always present and AUC-ROC is always defined.

    Args:
        labels: Patient labels of shape `(n_patients,)`.
        n_resamples: Number of resamples.
        seed: Seed of the generator.

    Returns:
        Patient indices of shape `(n_resamples, n_patients)`: the negative
        patients first, then the positive ones.

    Raises:
        ValueError: If a class has no patient.
    """
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Both classes need at least one patient to resample.")
    generator = np.random.default_rng(seed)
    return np.concatenate(
        [
            generator.choice(negative, (n_resamples, len(negative))),
            generator.choice(positive, (n_resamples, len(positive))),
        ],
        axis=1,
    )


def _metrics_of_runs(
    labels: np.ndarray, probabilities: np.ndarray, metrics: Sequence[str]
) -> dict[str, np.ndarray]:
    """Computes classification metrics for many runs at once.

    The same definitions as `MetricsAnalyzer.calculate_metrics`: positive when
    the probability is at least `THRESHOLD`, a ratio with a zero denominator is
    0, and AUC-ROC is the Mann-Whitney statistic with average ranks for ties.

    Args:
        labels: Labels of shape `(n,)`, both classes present.
        probabilities: Probabilities of shape `(n_runs, n)`.
        metrics: Names among accuracy, precision, recall, f1_score, and auc.

    Returns:
        Mapping of metric name to its value per run, shape `(n_runs,)`.
    """
    positive = labels == 1
    predicted = probabilities >= THRESHOLD
    true_positive = (predicted & positive).sum(axis=1)
    false_positive = (predicted & ~positive).sum(axis=1)
    false_negative = (~predicted & positive).sum(axis=1)
    true_negative = (~predicted & ~positive).sum(axis=1)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Divides, giving 0 where the denominator is 0."""
        return np.divide(
            numerator,
            denominator,
            out=np.zeros(len(numerator)),
            where=denominator > 0,
        )

    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    ranks = rankdata(probabilities, axis=1)
    values = {
        "accuracy": (true_positive + true_negative) / len(labels),
        "precision": ratio(true_positive, true_positive + false_positive),
        "recall": ratio(true_positive, true_positive + false_negative),
        "f1_score": ratio(
            2 * true_positive, 2 * true_positive + false_positive + false_negative
        ),
        "auc": (ranks[:, positive].sum(axis=1) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative),
    }
    return {metric: values[metric] for metric in metrics}


def _interval(values: np.ndarray) -> tuple[float, float]:
    """Returns the central `CONFIDENCE` percentile interval of the values."""
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_patients(
    payload: Mapping[str, Any],
    metrics: Sequence[str],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    other: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measures patient-level metrics and their spread over patients.

    For every run the slice probabilities are averaged per patient, the metric
    is computed over the patients, and the runs are averaged: the seed-mean
    patient-level metric. The patients are then resampled with replacement
    within each class, and the same seed-mean is computed on every resample.
    With `other`, both configurations are scored on the same resamples, which
    gives an interval for their difference.

    Args:
        payload: Results payload of the configuration, with `subject_ids`.
        metrics: Names of the metrics to compute.
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the resampling, so intervals are reproducible.
        other: Optional results payload of a second configuration, scored on
            the same patients.

    Returns:
        `patients` and `resamples` (the counts), `first` (per metric, `value`
        on all patients and the `CONFIDENCE` `interval` over resamples) and,
        with `other`, `second` and `difference` (`first - second`) in the same
        form.

    Raises:
        ValueError: If a payload has no subject IDs, or the two payloads were
            scored on different patients or labels.
    """
    patients, labels, probabilities = patient_probabilities(payload)
    resamples = stratified_resamples(labels, n_resamples, seed)
    sides = {"first": probabilities}
    if other is not None:
        other_patients, other_labels, other_probabilities = patient_probabilities(other)
        if other_patients != patients or not np.array_equal(other_labels, labels):
            raise ValueError(
                "The two configurations were scored on different patients."
            )
        sides["second"] = other_probabilities

    summary: dict[str, Any] = {"patients": len(patients), "resamples": n_resamples}
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for side, side_probabilities in sides.items():
        point = _metrics_of_runs(labels, side_probabilities, metrics)
        distributions[side] = {
            metric: np.array(
                [
                    _metrics_of_runs(
                        labels[row], side_probabilities[:, row], (metric,)
                    )[metric].mean()
                    for row in resamples
                ]
            )
            for metric in metrics
        }
        summary[side] = {
            metric: {
                "value": float(point[metric].mean()),
                "interval": _interval(distributions[side][metric]),
            }
            for metric in metrics
        }
    if other is not None:
        summary["difference"] = {
            metric: {
                "value": summary["first"][metric]["value"]
                - summary["second"][metric]["value"],
                "interval": _interval(
                    distributions["first"][metric] - distributions["second"][metric]
                ),
            }
            for metric in metrics
        }
    return summary


def _runs_by_seed(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """Indexes the runs of a results payload by their seed.

    Args:
        payload: A results payload.

    Returns:
        Mapping of seed to the run's entry.
    """
    seeds = seeds_of_runs(payload)
    return {seeds[run["run_id"]]: run for run in payload["individual_runs"]}
