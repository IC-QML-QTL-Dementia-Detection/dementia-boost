"""Human-readable lookup of runs and configurations.

Run and configuration IDs are hashes, so a person needs a way to see which
configuration and seed an ID belongs to. These functions build that table from
the training histories (which carry the full specs), and look an ID up by prefix.
Nothing here is stored: the table is rebuilt from the files on every call, so it
cannot go stale.
"""

import csv
import os
from collections.abc import Mapping, Sequence
from typing import Any

from dementia_boost.core.identity import config_id, label
from dementia_boost.core.layout import ResultsLayout, config_payload
from dementia_boost.telemetry.metrics import TrainingHistory
from dementia_boost.telemetry.run_listing import group_by_config

MIN_PREFIX_LENGTH = 4


def configuration_rows(runs: Sequence[TrainingHistory]) -> list[dict[str, Any]]:
    """Builds one table row per configuration.

    Args:
        runs: Histories, for example from `load_runs`.

    Returns:
        Rows with `paradigm`, `config_id`, `label`, `split_id`, `n_runs`, and
        the comma-separated `seeds`.
    """
    rows: list[dict[str, Any]] = []
    for configuration, members in group_by_config(list(runs)).items():
        first = members[0].spec
        rows.append(
            {
                "paradigm": first.paradigm.value,
                "config_id": configuration,
                "label": label(first, include_seed=False),
                "split_id": first.split_id,
                "n_runs": len(members),
                "seeds": ",".join(str(run.spec.seed) for run in members),
            }
        )
    return rows


def run_rows(
    runs: Sequence[TrainingHistory], layout: ResultsLayout
) -> list[dict[str, Any]]:
    """Builds one table row per run.

    Args:
        runs: Histories, for example from `load_runs`.
        layout: The layout that gives each run's file paths.

    Returns:
        Rows with `paradigm`, `config_id`, `seed`, `run_id`, `finished` (whether
        the checkpoint exists), and the `history` and `checkpoint` paths.
    """
    return [
        {
            "paradigm": run.spec.paradigm.value,
            "config_id": config_id(run.spec),
            "seed": run.spec.seed,
            "run_id": run.run_id,
            "finished": "yes" if layout.is_done(run.spec) else "no",
            "history": layout.history_path(run.spec),
            "checkpoint": layout.checkpoint_path(run.spec),
        }
        for run in runs
    ]


def find_matches(
    runs: Sequence[TrainingHistory], layout: ResultsLayout, prefix: str
) -> list[dict[str, Any]]:
    """Looks a configuration or run ID up by prefix.

    Args:
        runs: Histories, for example from `load_runs`.
        layout: The layout that gives file paths.
        prefix: The start of a `config_id` or `run_id`, at least
            `MIN_PREFIX_LENGTH` characters.

    Returns:
        One entry per matching configuration or run. A run entry has its full
        `spec` and `files`; a configuration entry has the spec without the seed,
        its `runs` (seed and run ID), and its `config.json` path.

    Raises:
        ValueError: If the prefix is shorter than `MIN_PREFIX_LENGTH`.
    """
    if len(prefix) < MIN_PREFIX_LENGTH:
        raise ValueError(
            f"Give at least {MIN_PREFIX_LENGTH} characters of the ID, got {prefix!r}."
        )

    matches: list[dict[str, Any]] = []
    for configuration, members in group_by_config(list(runs)).items():
        first = members[0].spec
        if configuration.startswith(prefix):
            payload = config_payload(first)
            matches.append(
                {
                    "kind": "configuration",
                    "id": configuration,
                    "label": payload["label"],
                    "spec": payload["spec"],
                    "runs": [
                        {"seed": run.spec.seed, "run_id": run.run_id} for run in members
                    ],
                    "files": {"config": layout.config_path(first)},
                }
            )
        for run in members:
            if run.run_id.startswith(prefix):
                matches.append(
                    {
                        "kind": "run",
                        "id": run.run_id,
                        "label": label(run.spec),
                        "spec": run.spec.to_dict(),
                        "files": {
                            "history": layout.history_path(run.spec),
                            "checkpoint": layout.checkpoint_path(run.spec),
                        },
                    }
                )
    return matches


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Formats rows as a plain-text table with padded columns.

    Args:
        rows: The rows to show.
        columns: Column names, in display order.

    Returns:
        A header, a separator, and one line per row; or a message when there are
        no rows.
    """
    if not rows:
        return "No runs found."

    cells = [[str(row[column]) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(line[i]) for line in cells))
        for i, column in enumerate(columns)
    ]

    def render(values: Sequence[str]) -> str:
        padded = (v.ljust(w) for v, w in zip(values, widths, strict=True))
        return "  ".join(padded).rstrip()

    lines = [render(columns), render(["-" * width for width in widths])]
    lines.extend(render(line) for line in cells)
    return "\n".join(lines)


def write_csv(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: str
) -> None:
    """Writes rows to a CSV file.

    Args:
        rows: The rows to write.
        columns: Column names, in order.
        path: Destination file, created with its directories if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
