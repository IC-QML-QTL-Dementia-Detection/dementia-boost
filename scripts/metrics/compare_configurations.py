"""Compares two configurations seed by seed on the test and validation cohorts.

For every seed both configurations ran, the difference `first - second` of each
metric is taken, and the script reports wins and losses, the median difference,
the Hodges-Lehmann estimate with a 95% interval, and the Wilcoxon signed-rank
p-value, raw and Holm-adjusted over the metrics (see
`dementia_boost.telemetry.statistics`). Test comes first; validation is the
development cohort. Every run is scored on the same patients, so this measures
the noise of the seed, not of the patients.

`--patients` adds the patient-level view: each run's slice probabilities are
averaged per patient, the seed-mean metric is computed over the patients, and a
class-stratified bootstrap over the patients gives a 95% interval for each
configuration and for their difference. It needs results files evaluated with
subject IDs.

    uv run scripts/metrics/compare_configurations.py FIRST_CONFIG SECOND_CONFIG
    uv run scripts/metrics/compare_configurations.py ed2f03b85c3f 62d37719d958 \\
        --root data/server --patients
"""

import argparse
import json
import os
import sys
from typing import Any

from dementia_boost.core.identity import Paradigm
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.report import REPORT_METRICS
from dementia_boost.telemetry.statistics import (
    DEFAULT_RESAMPLES,
    bootstrap_patients,
    compare_paired,
)

COHORTS: tuple[str, ...] = ("test", "val")


def load_results(layout: ResultsLayout, config: str, cohort: str) -> dict[str, Any]:
    """Loads a configuration's results on one cohort, whatever its paradigm.

    Args:
        layout: The results layout to read from.
        config: The configuration ID.
        cohort: `test` or `val`.

    Returns:
        The results payload.

    Raises:
        FileNotFoundError: If no paradigm has results for the configuration.
    """
    for paradigm in Paradigm:
        if config in layout.configs_with_metrics(paradigm, cohort):
            with open(layout.metrics_path(paradigm, config, cohort)) as file:
                return json.load(file)
    raise FileNotFoundError(
        f"No {cohort} results for configuration {config!r} under {layout.root}."
    )


def format_comparison(result: dict[str, Any]) -> str:
    """Renders one cohort's comparison as a Markdown table.

    Args:
        result: Output of `compare_paired`.

    Returns:
        A heading line and one table row per metric.
    """
    lines = [
        f"{result['cohort']}: {result['pairs']} paired seeds "
        f"(same initial weights: {result['same_initial_weights']})",
        "| Metric | Wins | Losses | Ties | Median diff | Hodges-Lehmann (95% CI) "
        "| p | p (Holm) |",
        "|" + " :-- |" * 8,
    ]
    for metric, row in result["metrics"].items():
        low, high = row["interval"]
        lines.append(
            f"| {metric} | {row['wins']} | {row['losses']} | {row['ties']} "
            f"| {row['median_difference']:+.4f} "
            f"| {row['hodges_lehmann']:+.4f} ({low:+.4f}, {high:+.4f}) "
            f"| {row['p_value']:.3g} | {row['p_value_holm']:.3g} |"
        )
    return "\n".join(lines)


def format_patients(cohort: str, result: dict[str, Any]) -> str:
    """Renders one cohort's patient-level comparison as a Markdown table.

    Args:
        cohort: The cohort name.
        result: Output of `bootstrap_patients` with a second configuration.

    Returns:
        A heading line and one table row per metric.
    """

    def cell(entry: dict[str, Any]) -> str:
        """Formats a value with its interval."""
        low, high = entry["interval"]
        return f"{entry['value']:.3f} ({low:.3f}, {high:.3f})"

    lines = [
        f"{cohort}, per patient: {result['patients']} patients, "
        f"{result['resamples']} resamples (95% intervals)",
        "| Metric | First | Second | Difference |",
        "|" + " :-- |" * 4,
    ]
    for metric in result["first"]:
        lines.append(
            f"| {metric} | {cell(result['first'][metric])} "
            f"| {cell(result['second'][metric])} "
            f"| {cell(result['difference'][metric])} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Compares the two configurations and writes the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", metavar="FIRST_CONFIG")
    parser.add_argument("second", metavar="SECOND_CONFIG")
    parser.add_argument(
        "--root",
        default=None,
        help="Results root to read from (default: the project's results).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON file to write (default: metrics/comparisons/ under the root).",
    )
    parser.add_argument(
        "--patients",
        action="store_true",
        help="Also compare per patient, with a bootstrap over the patients.",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    arguments = parser.parse_args()
    layout = ResultsLayout(arguments.root) if arguments.root else ResultsLayout()

    try:
        results: dict[str, Any] = {}
        for cohort in COHORTS:
            first = load_results(layout, arguments.first, cohort)
            second = load_results(layout, arguments.second, cohort)
            results[cohort] = compare_paired(first, second, REPORT_METRICS)
            if arguments.patients:
                results[cohort]["patients"] = bootstrap_patients(
                    first,
                    REPORT_METRICS,
                    n_resamples=arguments.resamples,
                    seed=arguments.bootstrap_seed,
                    other=second,
                )
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot compare: {error}", file=sys.stderr)
        sys.exit(1)

    output = arguments.output or os.path.join(
        layout.root,
        "metrics",
        "comparisons",
        f"{arguments.first}_vs_{arguments.second}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as file:
        json.dump(
            {"first": arguments.first, "second": arguments.second, **results},
            file,
            indent=2,
        )

    for cohort in COHORTS:
        print(format_comparison(results[cohort]), end="\n\n")
        if arguments.patients:
            print(format_patients(cohort, results[cohort]["patients"]), end="\n\n")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
