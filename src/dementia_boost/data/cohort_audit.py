"""Read-only audit of processed cohort directories for patient-level leakage.

The processed `.pt` files are named `<MRI ID>_<exam>.pt`, where the MRI ID is
`<Subject ID>_MR<k>`. The subject of a file is therefore recoverable from its
name, which lets this module report shared subjects without loading tensors.
"""

import os
import re

_SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+?)_MR\d+")


def subject_of(file_name: str) -> str:
    """Extracts the Subject ID from a processed tensor file name.

    Args:
        file_name: Base name such as `OAS2_0001_MR1_mpr-1.pt`.

    Returns:
        The Subject ID, for example `OAS2_0001`.

    Raises:
        ValueError: If the name does not contain a `_MR<k>` visit marker.
    """
    match = _SUBJECT_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Cannot infer a Subject ID from file name: {file_name}")
    return match.group("subject")


def list_cohort_subjects(cohort_dir: str) -> set[str]:
    """Lists the unique subjects that have a `.pt` file in a cohort directory.

    Args:
        cohort_dir: Directory holding the processed `.pt` files of one cohort.
            A missing directory is treated as an empty cohort.

    Returns:
        The set of Subject IDs found.
    """
    if not os.path.isdir(cohort_dir):
        return set()
    return {subject_of(name) for name in os.listdir(cohort_dir) if name.endswith(".pt")}


def find_shared_subjects(
    cohorts: dict[str, set[str]],
) -> dict[tuple[str, str], set[str]]:
    """Finds the subjects present in more than one cohort.

    Args:
        cohorts: Mapping of cohort name to its set of Subject IDs.

    Returns:
        A mapping from each pair of cohort names (sorted) to the Subject IDs
        they share. Pairs without a shared subject are omitted.
    """
    names = sorted(cohorts)
    shared: dict[tuple[str, str], set[str]] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            common = cohorts[first] & cohorts[second]
            if common:
                shared[(first, second)] = common
    return shared
