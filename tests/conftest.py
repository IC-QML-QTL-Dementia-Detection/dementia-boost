"""Shared test fixtures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from dementia_boost.data.split import SubjectSplit
from dementia_boost.data.split_manifest import (
    MANIFEST_NAME,
    build_manifest,
    write_manifest,
)

DEFAULT_COHORTS = {
    "train": [1, 2, 3, 4],
    "val": [5, 6],
    "test": [7, 8],
}
VISITS_PER_SUBJECT = 2


def subject_name(number: int) -> str:
    """Returns the canonical subject ID for a number, for example `OAS2_0003`."""
    return f"OAS2_{number:04d}"


@pytest.fixture
def make_layout() -> Callable[..., dict[str, Any]]:
    """Builds a consistent cohort layout and manifest under a directory.

    The label of a subject is its number modulo 2, so cohorts of two or more
    consecutive subjects hold both classes. Each subject gets two `.pt` files.

    Returns:
        A function `make(root, cohorts=None, image_size=8)` that writes the
        files and `split_manifest.json` under `root` and returns the manifest.
    """

    def make(
        root: Path,
        cohorts: dict[str, list[int]] | None = None,
        image_size: int = 8,
    ) -> dict[str, Any]:
        cohorts = cohorts or DEFAULT_COHORTS
        labels = {subject_name(n): n % 2 for ns in cohorts.values() for n in ns}
        counts = {subject: VISITS_PER_SUBJECT for subject in labels}
        for name, numbers in cohorts.items():
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            for number in numbers:
                subject = subject_name(number)
                for visit in range(1, VISITS_PER_SUBJECT + 1):
                    torch.save(
                        (torch.randn(1, image_size, image_size), labels[subject]),
                        directory / f"{subject}_MR{visit}_mpr-1.pt",
                    )
        split = SubjectSplit(
            train=[subject_name(n) for n in sorted(cohorts["train"])],
            val=[subject_name(n) for n in sorted(cohorts["val"])],
            test=[subject_name(n) for n in sorted(cohorts["test"])],
        )
        manifest = build_manifest(
            split=split,
            labels=labels,
            file_counts=counts,
            excluded={},
            skipped_files=[],
            seed=0,
            test_ratio=0.3,
            val_ratio=0.2,
            manual_train_ids=[],
            manual_test_ids=[],
        )
        write_manifest(str(root / MANIFEST_NAME), manifest)
        return manifest

    return make
