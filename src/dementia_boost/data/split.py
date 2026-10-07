"""Deterministic patient-level split into train, validation, and test cohorts.

The split is a pure function of its arguments. It sorts the subject IDs before
drawing, uses its own random generator, and never touches global RNG state, so
the same input and seed give the same assignment in every process and on every
machine.
"""

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

COHORT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SubjectSplit:
    """Subject IDs of the three cohorts, each sorted.

    Attributes:
        train: Subjects used to fit models.
        val: Subjects used for monitoring and selection.
        test: Subjects reported on once, never used to choose anything.
    """

    train: list[str]
    val: list[str]
    test: list[str]


def split_subjects(
    labels: Mapping[str, int],
    test_ratio: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 42,
    manual_train: Iterable[str] = (),
    manual_test: Iterable[str] = (),
) -> SubjectSplit:
    """Splits subjects into train, validation, and test cohorts.

    Order of operations: manual overrides are fixed first, then the test cohort
    is drawn from the remaining subjects, then the validation cohort is drawn
    from the subjects that are neither test nor forced into train. Everything
    left is train.

    Both draws are stratified by class, so every cohort keeps the class balance
    of the whole set. Per class, the test size is `round(n_class * test_ratio)`
    counting forced test subjects; if the manual test list is already larger,
    it is kept whole and nothing more is drawn. The validation size is
    `round(n_class_non_test * val_ratio)`, capped at the subjects eligible for
    it (forced-train subjects are not).

    Args:
        labels: Mapping of canonical subject ID to binary label.
        test_ratio: Share of all subjects placed in the test cohort.
        val_ratio: Share of the non-test subjects placed in the validation
            cohort.
        seed: Seed of the local generator used for both draws.
        manual_train: Subject IDs forced into train. They never enter val.
        manual_test: Subject IDs forced into test.

    Returns:
        The three sorted cohorts.

    Raises:
        ValueError: If an ID is in both override lists, or if a cohort ends up
            with fewer than two classes.
    """
    subjects = sorted(labels)
    forced_train = set(manual_train)
    forced_test = set(manual_test)

    both = forced_train & forced_test
    if both:
        raise ValueError(f"IDs forced into both train and test: {sorted(both)}")

    rng = random.Random(seed)
    classes = sorted(set(labels.values()))

    test = set(forced_test)
    for cls in classes:
        members = [s for s in subjects if labels[s] == cls]
        pool = [s for s in members if s not in forced_train | forced_test]
        target = round(len(members) * test_ratio)
        already = len([s for s in members if s in forced_test])
        test |= _draw(pool, target - already, rng)

    non_test = [s for s in subjects if s not in test]
    val: set[str] = set()
    for cls in classes:
        members = [s for s in non_test if labels[s] == cls]
        pool = [s for s in members if s not in forced_train]
        val |= _draw(pool, round(len(members) * val_ratio), rng)

    train = set(non_test) - val

    split = SubjectSplit(sorted(train), sorted(val), sorted(test))
    _require_both_classes(split, labels)
    return split


def _draw(pool: list[str], count: int, rng: random.Random) -> set[str]:
    """Draws up to `count` subjects from a sorted pool with the given generator.

    Args:
        pool: Eligible subject IDs, sorted.
        count: Number to draw. Zero or less draws nothing; more than the pool
            holds draws the whole pool.
        rng: The generator that owns the draw.

    Returns:
        The drawn subject IDs.
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return set(shuffled[: max(0, count)])


def _require_both_classes(split: SubjectSplit, labels: Mapping[str, int]) -> None:
    """Raises if any cohort of the split lacks one of the two classes.

    Args:
        split: The split to check.
        labels: Mapping of canonical subject ID to binary label.

    Raises:
        ValueError: If a cohort is empty or holds a single class.
    """
    for name, cohort in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        if len({labels[subject] for subject in cohort}) < 2:
            raise ValueError(
                f"The {name} cohort does not contain both classes "
                f"({len(cohort)} subjects). Change the seed or the ratios."
            )
