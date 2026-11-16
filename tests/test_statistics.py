"""Unit tests for the statistics computed over the runs of a configuration.

Regression coverage
-------------------
- The counts of collapsed runs, the majority-class accuracy, and the paired
  Wilcoxon comparisons cited in the results were computed by hand, outside the
  code that produces the results.
- Pairing two configurations by seed assumes the seed fixes the same initial
  weights on both sides, which holds only for heads of the same shape.
- Metrics over slices treat the correlated slices of one patient as
  independent cases, and the seed spread ignores which patients were drawn.
"""

import dataclasses
from dataclasses import asdict

import numpy as np
import pytest
import torch
from conftest import build_spec

from dementia_boost.core.identity import RunSpec, run_id
from dementia_boost.core.layout import config_payload
from dementia_boost.core.reproducibility import set_seed
from dementia_boost.models.quantum_cnn import QiskitQuantumClassifierHead
from dementia_boost.telemetry.metrics import MetricsAnalyzer
from dementia_boost.telemetry.statistics import (
    bootstrap_patients,
    compare_paired,
    count_collapsed_runs,
    holm_adjust,
    majority_class_accuracy,
    patient_probabilities,
    seeds_of_runs,
    share_initial_weights,
    stratified_resamples,
)

_LABELS = [0, 0, 0, 1, 1]
_METRICS = ("accuracy", "auc")


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


def _results(
    spec: RunSpec, values: dict[int, dict[str, float]], cohort: str = "test"
) -> dict:
    """Builds a results payload whose runs carry the real IDs of their seeds.

    Args:
        spec: Spec of any run of the configuration.
        values: Mapping of seed to that run's metrics.
        cohort: Cohort marker of the payload.

    Returns:
        A payload as `save_to_json` writes it, without the statistics.
    """
    return {
        "cohort": cohort,
        "configuration": config_payload(spec),
        "individual_runs": [
            {"run_id": run_id(dataclasses.replace(spec, seed=seed)), **metrics}
            for seed, metrics in values.items()
        ],
    }


def _shifted(shift: float, seeds: range = range(30)) -> dict[int, dict[str, float]]:
    """Metrics that vary by seed, plus a constant shift."""
    return {
        seed: {name: 0.5 + 0.01 * (seed % 7) + shift for name in _METRICS}
        for seed in seeds
    }


class TestSeedsOfRuns:
    """Validates that a run's seed is recovered from its ID and the spec."""

    def test_recovers_every_seed(self) -> None:
        """Each run ID maps back to the seed it was hashed with."""
        spec = build_spec("qiskit_qtl")
        payload = _results(spec, {3: {}, 0: {}, 57: {}})

        seeds = seeds_of_runs(payload)

        assert sorted(seeds.values()) == [0, 3, 57]
        assert seeds[run_id(dataclasses.replace(spec, seed=57))] == 57

    def test_a_run_of_another_configuration_is_refused(self) -> None:
        """A run ID that no seed of the configuration produces is an error."""
        payload = _results(build_spec("qiskit_qtl"), {1: {}})
        payload["individual_runs"].append({"run_id": "000000000000"})
        with pytest.raises(ValueError, match="000000000000"):
            seeds_of_runs(payload)


class TestComparePaired:
    """Validates the seed-by-seed comparison of two configurations."""

    def test_identical_configurations_show_no_difference(self) -> None:
        """Equal values for every seed give zero differences and p = 1."""
        spec = build_spec("qiskit_qtl")
        payload = _results(spec, _shifted(0.0))

        result = compare_paired(payload, payload, _METRICS)

        auc = result["metrics"]["auc"]
        assert (auc["wins"], auc["losses"], auc["ties"]) == (0, 0, 30)
        assert auc["hodges_lehmann"] == 0.0
        assert auc["p_value"] == 1.0

    def test_a_constant_shift_is_found_with_its_sign_and_size(self) -> None:
        """A configuration better by 0.05 in every seed wins every pair, with a
        Hodges-Lehmann estimate of 0.05 inside its interval and a small p."""
        better = _results(build_spec("qiskit_qtl", ansatz="rx"), _shifted(0.05))
        worse = _results(build_spec("qiskit_qtl"), _shifted(0.0))

        auc = compare_paired(better, worse, _METRICS)["metrics"]["auc"]

        assert auc["wins"] == 30
        assert auc["hodges_lehmann"] == pytest.approx(0.05)
        assert auc["interval"][0] - 1e-9 <= 0.05 <= auc["interval"][1] + 1e-9
        assert auc["p_value"] < 1e-5
        assert auc["p_value_holm"] >= auc["p_value"]

    def test_only_shared_seeds_are_paired(self) -> None:
        """Seeds present on one side only are counted, not paired."""
        first = _results(build_spec("qiskit_qtl"), _shifted(0.0, range(0, 20)))
        second = _results(
            build_spec("qiskit_qtl", ansatz="rx"), _shifted(0.0, range(10, 30))
        )

        result = compare_paired(first, second, _METRICS)

        assert result["pairs"] == 10
        assert (result["only_first"], result["only_second"]) == (10, 10)

    @pytest.mark.parametrize(
        "change",
        [
            {"spec": {"split_id": "other_split"}},
            {"cohort": "val"},
        ],
    )
    def test_incomparable_results_are_refused(self, change: dict) -> None:
        """Results of different splits or cohorts are not paired."""
        first = _results(build_spec("qiskit_qtl"), _shifted(0.0))
        second = _results(
            build_spec("qiskit_qtl", **change.get("spec", {})),
            _shifted(0.0),
            cohort=change.get("cohort", "test"),
        )
        with pytest.raises(ValueError):
            compare_paired(first, second, _METRICS)

    def test_no_shared_seed_is_refused(self) -> None:
        """Two configurations without a common seed cannot be paired."""
        first = _results(build_spec("qiskit_qtl"), _shifted(0.0, range(0, 5)))
        second = _results(
            build_spec("qiskit_qtl", ansatz="rx"), _shifted(0.0, range(5, 9))
        )
        with pytest.raises(ValueError, match="seed"):
            compare_paired(first, second, _METRICS)


def test_holm_adjustment_is_monotone_and_never_lowers_a_p_value() -> None:
    """Holm's step-down adjustment keeps the order of the p-values, multiplies
    the smallest by the number of tests, and caps at 1."""
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert holm_adjust({"a": 0.9, "b": 0.8})["a"] == 1.0


_PATIENT_METRICS = ("accuracy", "precision", "recall", "f1_score", "auc")


def _patient_results(
    probabilities: list[list[float]],
    labels: list[int],
    subjects: list[str] | None,
) -> dict:
    """Builds a results payload whose runs are scored like the evaluation does.

    Args:
        probabilities: One list of slice probabilities per run.
        labels: The slice labels, shared by every run.
        subjects: The subject of each slice, or None for an older file.

    Returns:
        A payload with `subject_ids` and runs carrying slice-level metrics.
    """
    runs = [
        asdict(
            MetricsAnalyzer.calculate_metrics(
                f"run_{index}", np.array(labels), np.array(probs)
            )
        )
        for index, probs in enumerate(probabilities)
    ]
    return {"cohort": "test", "subject_ids": subjects, "individual_runs": runs}


def _random_results(
    n_patients: int, shift: float, seed: int, slices_per_patient: int = 3
) -> dict:
    """Results of 5 runs on a balanced cohort, Demented probabilities shifted up."""
    generator = np.random.default_rng(seed)
    labels = [p % 2 for p in range(n_patients) for _ in range(slices_per_patient)]
    subjects = [
        f"OAS2_{p:04d}" for p in range(n_patients) for _ in range(slices_per_patient)
    ]
    probabilities = [
        list(
            np.clip(
                generator.uniform(0.2, 0.8, len(labels)) + shift * np.array(labels),
                0,
                1,
            )
        )
        for _ in range(5)
    ]
    return _patient_results(probabilities, labels, subjects)


class TestPatientLevel:
    """Validates metrics per patient and the bootstrap over patients."""

    def test_probabilities_are_averaged_per_patient(self) -> None:
        """Each patient gets the mean probability of its slices and its label."""
        payload = _patient_results(
            [[0.2, 0.4, 0.9, 0.7]],
            [0, 0, 1, 1],
            ["OAS2_0002", "OAS2_0002", "OAS2_0001", "OAS2_0001"],
        )

        patients, labels, probabilities = patient_probabilities(payload)

        assert patients == ["OAS2_0001", "OAS2_0002"]
        assert labels.tolist() == [1, 0]
        assert probabilities.shape == (1, 2)
        assert probabilities[0].tolist() == pytest.approx([0.8, 0.3])

    def test_one_slice_per_patient_gives_the_slice_metrics_back(self) -> None:
        """With one slice per patient, the patient-level metrics of each run are
        its stored slice-level metrics, so the vectorised metrics match
        scikit-learn's."""
        payload = _random_results(
            n_patients=20, shift=0.1, seed=0, slices_per_patient=1
        )

        result = bootstrap_patients(payload, _PATIENT_METRICS, n_resamples=10, seed=0)

        for metric in _PATIENT_METRICS:
            stored = np.mean([run[metric] for run in payload["individual_runs"]])
            assert result["first"][metric]["value"] == pytest.approx(stored)
        assert result["patients"] == 20

    def test_identical_configurations_differ_by_exactly_zero(self) -> None:
        """Paired resamples of the same results give a zero difference."""
        payload = _random_results(n_patients=20, shift=0.2, seed=0)

        result = bootstrap_patients(
            payload, _PATIENT_METRICS, n_resamples=50, seed=0, other=payload
        )

        assert result["difference"]["auc"]["interval"] == (0.0, 0.0)

    def test_a_large_gap_gives_an_interval_that_excludes_zero(self) -> None:
        """A configuration that separates the classes far better than another
        keeps a positive AUC difference on every resample of patients."""
        strong = _random_results(n_patients=30, shift=0.6, seed=0)
        weak = _random_results(n_patients=30, shift=0.0, seed=1)

        result = bootstrap_patients(
            strong, _PATIENT_METRICS, n_resamples=200, seed=0, other=weak
        )

        low, high = result["difference"]["auc"]["interval"]
        assert 0.0 < low <= high

    def test_every_resample_keeps_both_classes_and_their_sizes(self) -> None:
        """Resampling within each class keeps the class counts of the cohort."""
        labels = np.array([0] * 7 + [1] * 3)

        resamples = stratified_resamples(labels, n_resamples=100, seed=0)

        assert resamples.shape == (100, 10)
        assert all(labels[row].sum() == 3 for row in resamples)

    def test_the_seed_fixes_the_intervals(self) -> None:
        """The bootstrap has its own seed, so intervals are reproducible."""
        payload = _random_results(n_patients=20, shift=0.2, seed=0)

        first = bootstrap_patients(payload, ("auc",), n_resamples=50, seed=3)
        again = bootstrap_patients(payload, ("auc",), n_resamples=50, seed=3)

        assert first == again

    def test_results_without_subjects_are_refused(self) -> None:
        """Older results files have no subjects; the message says what to do."""
        payload = _patient_results([[0.2, 0.8]], [0, 1], None)
        with pytest.raises(ValueError, match="evaluation"):
            bootstrap_patients(payload, ("auc",))

    def test_a_patient_with_two_labels_is_refused(self) -> None:
        """A subject's diagnosis is constant, so mixed labels mean bad input."""
        payload = _patient_results([[0.2, 0.8]], [0, 1], ["OAS2_0001", "OAS2_0001"])
        with pytest.raises(ValueError, match="OAS2_0001"):
            patient_probabilities(payload)

    def test_configurations_scored_on_other_patients_are_refused(self) -> None:
        """A paired difference needs the same patients on both sides."""
        first = _random_results(n_patients=10, shift=0.2, seed=0)
        second = _random_results(n_patients=12, shift=0.2, seed=0)
        with pytest.raises(ValueError, match="patients"):
            bootstrap_patients(first, ("auc",), other=second)


def _initial_head(seed: int, **options) -> QiskitQuantumClassifierHead:
    """Builds a head as the training script does: seed, construct, Glorot."""
    set_seed(seed)
    head = QiskitQuantumClassifierHead(
        in_features=16, n_qubits=3, n_layers=2, seed=seed, **options
    )
    head.apply(QiskitQuantumClassifierHead.apply_glorot_init)
    return head


class TestPairingAssumption:
    """Checks when the same seed gives both configurations the same start."""

    def test_same_block_different_embedding_starts_identically(self) -> None:
        """The embedding has no weights, so one seed gives identical pre-net,
        post-net, and circuit weights."""
        first = _initial_head(7, embedding="paper").state_dict()
        second = _initial_head(7, embedding="zz_feature_map").state_dict()

        assert first.keys() == second.keys()
        assert all(torch.equal(first[key], second[key]) for key in first)
        assert share_initial_weights(
            build_spec("qiskit_qtl"), build_spec("qiskit_qtl", ansatz="rx")
        )

    def test_a_different_block_shifts_every_later_draw(self) -> None:
        """Another block has another number of circuit weights, so the draws
        that follow it, the classical initialisation included, differ."""
        first = _initial_head(7).state_dict()
        second = _initial_head(7, block="trainable_zz").state_dict()

        assert not torch.equal(first["pre_net.weight"], second["pre_net.weight"])
        assert not share_initial_weights(
            build_spec("qiskit_qtl"),
            build_spec("qiskit_qtl", block="trainable_zz"),
        )
