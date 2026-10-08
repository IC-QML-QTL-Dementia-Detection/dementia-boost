"""Shared test fixtures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from dementia_boost.core.identity import Paradigm, RunSpec
from dementia_boost.data.split import SubjectSplit
from dementia_boost.data.split_manifest import (
    MANIFEST_NAME,
    build_manifest,
    write_manifest,
)

_SPEC_DEFAULTS: dict[str, Any] = {
    "lr": 1e-3,
    "lr_step_size": 5,
    "lr_gamma": 0.5,
    "epochs": 2,
    "batch_size": 4,
    "split_id": "5192b1b7c0d3",
    "seed": 1,
}
_HEAD_DEFAULTS: dict[Paradigm, dict[str, Any]] = {
    Paradigm.BASELINE: {},
    Paradigm.CTL: {"backbone_id": "c8375944e970"},
    Paradigm.PL_QTL: {
        "ansatz": "paper",
        "n_qubits": 2,
        "n_layers": 1,
        "gradient": "adjoint",
        "backbone_id": "c8375944e970",
    },
    Paradigm.QISKIT_QTL: {
        "ansatz": "paper",
        "n_qubits": 2,
        "n_layers": 1,
        "gradient": "spsa",
        "spsa_epsilon": 0.1,
        "backbone_id": "c8375944e970",
    },
}


def build_spec(
    paradigm: Paradigm | str = Paradigm.BASELINE, **overrides: Any
) -> RunSpec:
    """Builds a valid `RunSpec` with small test defaults.

    Args:
        paradigm: The paradigm; its head fields get valid defaults.
        **overrides: Fields to replace.

    Returns:
        The validated spec.
    """
    paradigm = Paradigm(paradigm)
    return RunSpec(
        **{
            "paradigm": paradigm,
            **_SPEC_DEFAULTS,
            **_HEAD_DEFAULTS[paradigm],
            **overrides,
        }
    )


@pytest.fixture
def make_spec() -> Callable[..., RunSpec]:
    """Returns `build_spec`, for tests that need valid run specs."""
    return build_spec


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
