"""Read-only audit of the processed cohorts for patient-level leakage.

Lists the subjects of each cohort directory under `data/results` and reports
every subject that appears in more than one. Exits with status 1 on a leak, so
it can gate a pipeline. Nothing on disk is modified.
"""

import sys

from dementia_boost.data.cohort_audit import find_shared_subjects, list_cohort_subjects
from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.split import COHORT_NAMES


def main() -> None:
    """Prints subjects per cohort and the shared subjects, if any."""
    root = OasisDataLoader.RESULTS_PATH
    cohorts = {name: list_cohort_subjects(f"{root}/{name}") for name in COHORT_NAMES}

    for name, subjects in cohorts.items():
        print(f"{name}: {len(subjects)} subjects")

    shared = find_shared_subjects(cohorts)
    if not shared:
        print("No subject appears in more than one cohort.")
        return

    for (first, second), subjects in shared.items():
        print(f"LEAK {first} / {second}: {len(subjects)} shared: {sorted(subjects)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
