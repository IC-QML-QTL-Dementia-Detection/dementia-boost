"""Integrity guard that checks the cohort files on disk against the manifest.

The manifest says which subject belongs to which cohort and how many files each
subject has. This module compares that record with what is actually in the
cohort directories, so a stale, partial, or hand-edited layout is refused before
anything is trained on it.
"""

import os
from collections import Counter
from typing import Any

from dementia_boost.data.cohort_audit import find_shared_subjects, subject_of
from dementia_boost.data.split import COHORT_NAMES
from dementia_boost.data.split_manifest import MANIFEST_NAME, read_manifest


class SplitIntegrityError(ValueError):
    """Raised when the cohort files on disk do not match the split manifest."""


def verify_split_layout(results_dir: str) -> dict[str, Any]:
    """Checks the cohort directories against `split_manifest.json`.

    The checks are: no subject appears in two cohort directories; every file
    sits in the cohort the manifest assigns to its subject; every subject has
    exactly the number of files the manifest records (so missing and extra
    files are both caught); and every cohort holds both classes. All problems
    are collected and reported together, with the offending subject IDs.

    Args:
        results_dir: Directory holding `train/`, `val/`, `test/` and the
            manifest.

    Returns:
        The manifest, so callers can read the `split_id` without a second read.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        SplitIntegrityError: If any check fails.
    """
    manifest = read_manifest(os.path.join(results_dir, MANIFEST_NAME))
    expected = manifest["subjects"]

    files_on_disk: dict[str, Counter[str]] = {}
    problems: list[str] = []
    for cohort in COHORT_NAMES:
        counter: Counter[str] = Counter()
        directory = os.path.join(results_dir, cohort)
        names = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
        for name in (n for n in names if n.endswith(".pt")):
            try:
                counter[subject_of(name)] += 1
            except ValueError:
                problems.append(f"unrecognised file name in {cohort}: {name}")
        files_on_disk[cohort] = counter

    shared = find_shared_subjects({c: set(n) for c, n in files_on_disk.items()})
    for (first, second), subjects in shared.items():
        problems.append(f"subjects in both {first} and {second}: {sorted(subjects)}")

    for cohort, counter in files_on_disk.items():
        for subject in sorted(counter):
            if subject not in expected:
                problems.append(f"{subject} in {cohort} is not in the manifest")
            elif expected[subject]["cohort"] != cohort:
                problems.append(
                    f"{subject} is in {cohort} but the manifest assigns it to "
                    f"{expected[subject]['cohort']}"
                )

    for subject in sorted(expected):
        cohort = expected[subject]["cohort"]
        found = files_on_disk[cohort][subject]
        if found != expected[subject]["n_files"]:
            problems.append(
                f"{subject} has {found} files in {cohort}, "
                f"the manifest records {expected[subject]['n_files']}"
            )

    for cohort in COHORT_NAMES:
        balance = manifest["cohorts"][cohort]["class_balance"]
        if len(balance) < 2 or min(balance.values()) == 0:
            problems.append(
                f"the {cohort} cohort does not hold both classes: {balance}"
            )

    if problems:
        raise SplitIntegrityError(
            f"Cohorts in {results_dir} do not match the split manifest "
            f"(split_id {manifest['split_id']}):\n- " + "\n- ".join(problems)
        )
    return manifest
