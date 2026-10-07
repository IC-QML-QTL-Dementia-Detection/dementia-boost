"""Split manifest: the single record of which patient sits in which cohort.

The manifest is written next to the processed cohorts. It holds no timestamps
and uses sorted keys, so running the ETL twice with the same inputs produces a
byte-identical file.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from dementia_boost.core.identity import short_hash
from dementia_boost.data.split import SubjectSplit

MANIFEST_NAME = "split_manifest.json"


def compute_split_id(split: SubjectSplit) -> str:
    """Hashes the subject-to-cohort assignment into a short identifier.

    Only the assignment is hashed, not the seed, the ratios, or the labels, so
    the ID changes when and only when a subject changes cohort.

    Args:
        split: The three sorted cohorts.

    Returns:
        The short hash of the assignment (see `short_hash`).
    """
    return short_hash({"train": split.train, "val": split.val, "test": split.test})


def build_manifest(
    split: SubjectSplit,
    labels: Mapping[str, int],
    file_counts: Mapping[str, int],
    excluded: Mapping[str, str],
    skipped_files: Sequence[Mapping[str, str]],
    seed: int,
    test_ratio: float,
    val_ratio: float,
    manual_train_ids: Sequence[str],
    manual_test_ids: Sequence[str],
) -> dict[str, Any]:
    """Assembles the manifest of one ETL run.

    Args:
        split: The three sorted cohorts.
        labels: Canonical subject ID to binary label, for the split subjects.
        file_counts: Canonical subject ID to the number of `.pt` files written.
        excluded: Canonical subject ID (or folder name) to the reason it was
            left out.
        skipped_files: Raw files that were not used, each with `file` and
            `reason`.
        seed: Seed given to the split.
        test_ratio: Test ratio given to the split.
        val_ratio: Validation ratio given to the split.
        manual_train_ids: Canonical IDs forced into train.
        manual_test_ids: Canonical IDs forced into test.

    Returns:
        A JSON-serialisable dictionary.
    """
    cohorts = {"train": split.train, "val": split.val, "test": split.test}
    return {
        "split_id": compute_split_id(split),
        "seed": seed,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "manual_train_ids": list(manual_train_ids),
        "manual_test_ids": list(manual_test_ids),
        "cohorts": {
            name: {
                "n_subjects": len(subjects),
                "n_files": sum(file_counts[s] for s in subjects),
                "class_balance": {
                    str(cls): sum(1 for s in subjects if labels[s] == cls)
                    for cls in sorted(set(labels.values()))
                },
            }
            for name, subjects in cohorts.items()
        },
        "subjects": {
            subject: {
                "cohort": name,
                "label": labels[subject],
                "n_files": file_counts[subject],
            }
            for name, subjects in cohorts.items()
            for subject in subjects
        },
        "excluded": dict(excluded),
        "skipped_files": [dict(item) for item in skipped_files],
    }


def read_manifest(path: str) -> dict[str, Any]:
    """Reads a manifest written by `write_manifest`.

    Args:
        path: Manifest file path.

    Returns:
        The manifest dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{MANIFEST_NAME} not found at {path}. "
            "Run the ETL (scripts/etl_pipeline.py) to create the cohorts."
        )
    with open(path) as handle:
        return json.load(handle)


def write_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    """Writes the manifest atomically, through a temporary file.

    Args:
        path: Destination file path.
        manifest: The manifest to serialise.
    """
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)
