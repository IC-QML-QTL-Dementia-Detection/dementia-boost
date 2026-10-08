"""Unit tests for the deterministic three-way subject split.

Regression coverage
-------------------
- The split depended on the per-process string hash seed, because the subject
  set was shuffled without sorting first.
- The split read and reseeded the global NumPy RNG.
- Nothing checked that the cohorts were disjoint or contained both classes.
"""

import itertools
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from dementia_boost.data.split import SubjectSplit, split_subjects


def _labels(n: int) -> dict[str, int]:
    """Builds n subjects with alternating labels."""
    return {f"OAS2_{i:04d}": i % 2 for i in range(1, n + 1)}


class TestPartition:
    """Validates disjointness, coverage, and ordering."""

    @pytest.mark.parametrize(
        ("seed", "test_ratio", "val_ratio"),
        list(itertools.product([0, 1, 42, 99], [0.2, 0.3], [0.1, 0.2])),
    )
    def test_cohorts_are_disjoint_and_cover_all_subjects(
        self, seed: int, test_ratio: float, val_ratio: float
    ) -> None:
        """Every subject lands in exactly one cohort."""
        labels = _labels(40)
        split = split_subjects(labels, test_ratio, val_ratio, seed)
        parts = [set(split.train), set(split.val), set(split.test)]
        assert sum(len(p) for p in parts) == len(labels)
        assert set().union(*parts) == set(labels)

    def test_cohorts_are_sorted_lists(self) -> None:
        """Outputs are sorted so they can be hashed and compared directly."""
        split = split_subjects(_labels(30), 0.3, 0.2, 7)
        for cohort in (split.train, split.val, split.test):
            assert cohort == sorted(cohort)

    def test_sizes_follow_the_ratios(self) -> None:
        """Test is 30% of all subjects, val is 20% of the non-test subjects."""
        split = split_subjects(_labels(100), 0.3, 0.2, 0)
        assert len(split.test) == 30
        assert len(split.val) == 14
        assert len(split.train) == 56

    def test_both_classes_in_every_cohort(self) -> None:
        """Every cohort of a normal split holds both classes."""
        labels = _labels(40)
        split = split_subjects(labels, 0.3, 0.2, 3)
        for cohort in (split.train, split.val, split.test):
            assert {labels[s] for s in cohort} == {0, 1}

    def test_one_class_cohort_raises(self) -> None:
        """A cohort with a single class is rejected, not silently accepted."""
        labels = {f"OAS2_{i:04d}": 0 for i in range(1, 21)}
        with pytest.raises(ValueError, match="class"):
            split_subjects(labels, 0.3, 0.2, 0)


class TestDeterminism:
    """Validates independence from process state."""

    def test_same_seed_same_split(self) -> None:
        """Two calls with identical arguments agree."""
        assert split_subjects(_labels(30), 0.3, 0.2, 5) == split_subjects(
            _labels(30), 0.3, 0.2, 5
        )

    def test_different_seed_changes_split(self) -> None:
        """The seed actually drives the draw."""
        assert split_subjects(_labels(30), 0.3, 0.2, 1) != split_subjects(
            _labels(30), 0.3, 0.2, 2
        )

    def test_input_order_does_not_matter(self) -> None:
        """The same subjects in another dict order give the same split."""
        labels = _labels(30)
        shuffled = dict(random.Random(0).sample(sorted(labels.items()), len(labels)))
        assert split_subjects(labels, 0.3, 0.2, 5) == split_subjects(
            shuffled, 0.3, 0.2, 5
        )

    def test_global_rng_is_untouched(self) -> None:
        """The split neither reads nor reseeds the global NumPy or random RNG."""
        np.random.seed(123)
        random.seed(123)
        expected_numpy = np.random.rand()
        expected_python = random.random()
        np.random.seed(123)
        random.seed(123)

        split_subjects(_labels(30), 0.3, 0.2, 5)

        assert np.random.rand() == expected_numpy
        assert random.random() == expected_python

    def test_identical_across_hash_seeds(self) -> None:
        """The split is the same in processes with different PYTHONHASHSEED."""
        script = textwrap.dedent(
            """
            from dementia_boost.data.split import split_subjects

            labels = {f"OAS2_{i:04d}": i % 2 for i in range(1, 41)}
            split = split_subjects(labels, 0.3, 0.2, 42)
            print(split.train, split.val, split.test)
            """
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                env={"PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
            ).stdout
            for hash_seed in ("1", "2", "3")
        }
        assert len(outputs) == 1


class TestOverrides:
    """Validates that manual overrides are applied before the ratios."""

    def test_manual_train_ids_stay_in_train_never_in_val(self) -> None:
        """Forced-train subjects are never drawn into val or test."""
        forced = ["OAS2_0001", "OAS2_0002", "OAS2_0003"]
        for seed in range(20):
            split = split_subjects(_labels(40), 0.3, 0.2, seed, manual_train=forced)
            assert set(forced) <= set(split.train)

    def test_manual_test_ids_stay_in_test(self) -> None:
        """Forced-test subjects end up in test."""
        forced = ["OAS2_0004", "OAS2_0005"]
        split = split_subjects(_labels(40), 0.3, 0.2, 1, manual_test=forced)
        assert set(forced) <= set(split.test)
        assert len(split.test) == 12

    def test_manual_test_larger_than_target_keeps_all_of_it(self) -> None:
        """An override above the test target is kept whole and nothing else drawn."""
        forced = [f"OAS2_{i:04d}" for i in range(1, 9)]
        split = split_subjects(_labels(20), 0.2, 0.2, 0, manual_test=forced)
        assert set(split.test) == set(forced)

    def test_overlapping_overrides_raise(self) -> None:
        """An ID in both override lists is rejected."""
        with pytest.raises(ValueError, match="OAS2_0001"):
            split_subjects(
                _labels(20),
                0.3,
                0.2,
                0,
                manual_train=["OAS2_0001"],
                manual_test=["OAS2_0001"],
            )


def test_split_dataclass_holds_three_cohorts() -> None:
    """`SubjectSplit` exposes the three cohorts by name."""
    split = SubjectSplit(train=["a"], val=["b"], test=["c"])
    assert (split.train, split.val, split.test) == (["a"], ["b"], ["c"])
