"""Discovery of runs from their training history files.

A history file carries the full spec of its run, so the runs of a configuration
are found by reading histories, never by parsing file names. A history that
sits anywhere other than where its own spec places it is refused.
"""

import os

from dementia_boost.core.identity import Paradigm, config_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import MetricsAnalyzer, TrainingHistory


def load_runs(
    layout: ResultsLayout,
    paradigm: Paradigm | str | None = None,
    configuration: str | None = None,
) -> list[TrainingHistory]:
    """Loads the training histories of the runs on disk.

    Args:
        layout: The results layout to read from.
        paradigm: Restrict to one paradigm. Defaults to all.
        configuration: Restrict to one `config_id`. Defaults to all.

    Returns:
        The histories sorted by paradigm, configuration ID, and seed.

    Raises:
        ValueError: If a history file is not at the path its own spec gives
            (moved or renamed by hand), or its stored run ID does not match.
    """
    runs: list[TrainingHistory] = []
    for path in layout.history_files(paradigm, configuration):
        history = MetricsAnalyzer.load_history(path)
        expected = layout.history_path(history.spec)
        if os.path.abspath(expected) != os.path.abspath(path):
            raise ValueError(
                f"History {path} does not belong at this location; "
                f"its spec places it at {expected}."
            )
        runs.append(history)

    return sorted(
        runs,
        key=lambda run: (run.spec.paradigm.value, config_id(run.spec), run.spec.seed),
    )


def group_by_config(runs: list[TrainingHistory]) -> dict[str, list[TrainingHistory]]:
    """Groups runs by configuration, so the seeds of one configuration sit together.

    Args:
        runs: Histories, for example from `load_runs`.

    Returns:
        A mapping from `config_id` to its runs sorted by seed.
    """
    groups: dict[str, list[TrainingHistory]] = {}
    for run in runs:
        groups.setdefault(config_id(run.spec), []).append(run)
    return {
        key: sorted(members, key=lambda run: run.spec.seed)
        for key, members in groups.items()
    }
