"""Read-only lookup table of the runs and configurations on disk.

Run and configuration IDs are hashes. This script rebuilds, on every call, a
table that says which configuration and seed each ID belongs to, from the
specs stored in the training histories. Nothing is modified.

    uv run scripts/list_runs.py                      one row per configuration
    uv run scripts/list_runs.py --runs               one row per run
    uv run scripts/list_runs.py --paradigm pl_qtl    only one paradigm
    uv run scripts/list_runs.py --find 62753438      look an ID (prefix) up
    uv run scripts/list_runs.py --runs --csv out.csv export the table
"""

import argparse
import json
import sys

from dementia_boost.core.identity import Paradigm
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.run_listing import load_runs
from dementia_boost.telemetry.run_table import (
    configuration_rows,
    find_matches,
    format_table,
    run_rows,
    write_csv,
)

CONFIGURATION_COLUMNS = [
    "paradigm",
    "config_id",
    "label",
    "split_id",
    "n_runs",
    "seeds",
]
RUN_COLUMNS = [
    "paradigm",
    "config_id",
    "seed",
    "run_id",
    "finished",
    "history",
    "checkpoint",
]


def main() -> None:
    """Prints the lookup table, a lookup result, or writes the table to CSV."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--runs", action="store_true", help="one row per run instead of per config"
    )
    parser.add_argument(
        "--paradigm",
        choices=[paradigm.value for paradigm in Paradigm],
        help="only this paradigm",
    )
    parser.add_argument(
        "--find", metavar="PREFIX", help="look up a config or run ID by prefix"
    )
    parser.add_argument("--csv", metavar="PATH", help="also write the table to CSV")
    arguments = parser.parse_args()

    layout = ResultsLayout()
    try:
        runs = load_runs(layout, arguments.paradigm)

        if arguments.find:
            matches = find_matches(runs, layout, arguments.find)
            if not matches:
                print(f"No configuration or run matches {arguments.find!r}.")
                sys.exit(1)
            print(json.dumps(matches, indent=2))
            return

        columns = RUN_COLUMNS if arguments.runs else CONFIGURATION_COLUMNS
        rows = run_rows(runs, layout) if arguments.runs else configuration_rows(runs)
    except ValueError as error:
        print(f"Cannot list the runs: {error}")
        sys.exit(1)

    print(format_table(rows, columns))
    if arguments.csv:
        write_csv(rows, columns, arguments.csv)
        print(f"\nWritten to {arguments.csv}")


if __name__ == "__main__":
    main()
