"""Canonical subject IDs, label mapping, and manual override validation.

OASIS-II spells the same subject in several ways across its sources: the CSV
uses `OAS2_NNNN`, raw folders use `OAS2_NNNN_MRk`, and labels differ in case.
A subject is a patient and the unit of the split. A visit (`OAS2_NNNN_MRk`) is
one exam of that patient; it keeps its own file and is never merged with other
visits. Every boundary of the pipeline goes through this module, so one subject
has one ID and one label everywhere.
"""

import re
from collections.abc import Collection, Iterable

import pandas as pd

_SUBJECT_PATTERN = re.compile(r"^OAS2_\d{4}$")
_VISIT_PATTERN = re.compile(r"^(OAS2_\d{4})_MR\d+$")
_LABELS: dict[str, int] = {"nondemented": 0, "demented": 1}
_CONVERTED = "converted"


def canonical_subject_id(raw: str) -> str:
    """Normalises a subject ID to the canonical `OAS2_NNNN` form.

    Args:
        raw: A Subject ID, in any case, with optional surrounding whitespace.

    Returns:
        The uppercase Subject ID.

    Raises:
        ValueError: If the value is not `OAS2_` plus four digits after
            normalisation. Visit IDs (`OAS2_NNNN_MRk`) are rejected, because
            they name one exam and not a patient.
    """
    normalised = str(raw).strip().upper()
    if _SUBJECT_PATTERN.match(normalised) is None:
        raise ValueError(f"Malformed OASIS-II subject ID: {raw!r}")
    return normalised


def subject_of_visit(raw: str) -> str:
    """Maps a visit (MRI ID or raw folder name) to the subject it belongs to.

    Args:
        raw: A visit ID such as `OAS2_0001_MR2`, in any case.

    Returns:
        The canonical Subject ID, for example `OAS2_0001`.

    Raises:
        ValueError: If the value is not `OAS2_NNNN_MRk`.
    """
    match = _VISIT_PATTERN.match(str(raw).strip().upper())
    if match is None:
        raise ValueError(f"Malformed OASIS-II visit ID: {raw!r}")
    return match.group(1)


def canonical_label(raw: str) -> int | None:
    """Maps a diagnostic group name to its binary class.

    Args:
        raw: The `Group` value, in any case and with optional whitespace.

    Returns:
        0 for Nondemented, 1 for Demented, and None for Converted, which is
        excluded by design.

    Raises:
        ValueError: If the group name is not recognised.
    """
    key = str(raw).strip().lower()
    if key == _CONVERTED:
        return None
    if key not in _LABELS:
        raise ValueError(f"Unknown diagnostic group: {raw!r}")
    return _LABELS[key]


def build_subject_labels(
    frame: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, str]]:
    """Builds one label per subject from the visit-level metadata.

    The label is a property of the patient, so the visit rows of a subject
    collapse into one entry here. The visits themselves are not tracked by this
    mapping; they are found on disk when the subject's cohort is written.

    Args:
        frame: Metadata with `Subject ID` and `Group` columns, one row per visit.

    Returns:
        A pair of mappings: canonical subject ID to binary label, and canonical
        subject ID to the reason it was excluded.

    Raises:
        ValueError: If a subject has conflicting groups, or an ID or group is
            malformed.
    """
    groups: dict[str, set[int | None]] = {}
    for subject, group in zip(frame["Subject ID"], frame["Group"], strict=True):
        groups.setdefault(canonical_subject_id(subject), set()).add(
            canonical_label(group)
        )

    labels: dict[str, int] = {}
    excluded: dict[str, str] = {}
    for subject in sorted(groups):
        found = groups[subject]
        if len(found) > 1:
            raise ValueError(f"Subject {subject} has conflicting groups: {found}")
        (label,) = found
        if label is None:
            excluded[subject] = "Converted group is excluded from the binary task"
        else:
            labels[subject] = label
    return labels, excluded


def validate_overrides(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    known_subjects: Collection[str],
) -> tuple[list[str], list[str]]:
    """Canonicalises and validates the manual train and test overrides.

    Args:
        train_ids: Subject IDs forced into the training cohort.
        test_ids: Subject IDs forced into the test cohort.
        known_subjects: Canonical IDs of the labelled subjects.

    Returns:
        The sorted canonical train and test override lists.

    Raises:
        ValueError: If an ID is malformed (visit IDs included), unknown, or
            appears in both lists.
    """
    train = {canonical_subject_id(subject) for subject in train_ids}
    test = {canonical_subject_id(subject) for subject in test_ids}

    both = train & test
    if both:
        raise ValueError(f"IDs forced into both train and test: {sorted(both)}")
    unknown = (train | test) - set(known_subjects)
    if unknown:
        raise ValueError(f"Override IDs not among the known subjects: {unknown}")
    return sorted(train), sorted(test)
