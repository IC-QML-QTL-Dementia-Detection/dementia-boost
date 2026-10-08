"""Comparative improvement report generator across paradigms on NIfTI data.

This script aggregates evaluation results across the Classical Baseline, Classical
Transfer Learning (CTL), PennyLane Quantum Transfer Learning (QTL), and Qiskit
Quantum Transfer Learning (Qiskit QTL) runs on NIfTI data. It reports the mean
and standard deviation of every metric across seeds on the test cohort, shows
the test metrics of the run selected on the validation cohort, calculates the
relative percentage change between the means, and exports a comparative JSON
report. No run is chosen on test metrics, and paradigms trained on different
splits are refused.

Each paradigm contributes one configuration: the only one with results, or the
one named with `--config <paradigm>=<config_id>` when there are several.
"""

import argparse
import json
import os
import sys

from dementia_boost.core.identity import Paradigm
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.report import (
    build_comparative_report,
    resolve_configuration,
)

REQUIRED_PARADIGMS = (Paradigm.BASELINE, Paradigm.CTL)


def parse_requested(items: list[str]) -> dict[str, str]:
    """Parses repeated `--config paradigm=config_id` arguments.

    Args:
        items: Arguments of the form `<paradigm>=<config_id>`.

    Returns:
        Mapping of paradigm name to the requested configuration ID.

    Raises:
        ValueError: If an item is not of that form or names an unknown paradigm.
    """
    requested: dict[str, str] = {}
    for item in items:
        name, separator, configuration = item.partition("=")
        if not separator or not configuration:
            raise ValueError(f"Expected <paradigm>=<config_id>, got {item!r}.")
        requested[Paradigm(name).value] = configuration
    return requested


def main() -> None:
    """Generates the quantitative comparative improvement report for NIfTI."""
    logger = setup_logger("improvement_report_nifti")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PARADIGM=CONFIG_ID",
        help="Configuration to report for a paradigm with several (repeatable).",
    )
    arguments = parser.parse_args()

    layout = ResultsLayout()
    test_results: dict[str, dict] = {}
    val_results: dict[str, dict] = {}

    try:
        requested = parse_requested(arguments.config)
        for paradigm in Paradigm:
            configuration = resolve_configuration(
                layout, paradigm, requested.get(paradigm.value)
            )
            if configuration is None:
                if paradigm in REQUIRED_PARADIGMS:
                    logger.error(f"Missing telemetry for {paradigm.value}.")
                    sys.exit(1)
                logger.info(f"Skipping {paradigm.value}: no results.")
                continue
            for cohort, store in (("test", test_results), ("val", val_results)):
                with open(layout.metrics_path(paradigm, configuration, cohort)) as file:
                    store[paradigm.value] = json.load(file)

        report = build_comparative_report(test_results, val_results)
    except ValueError as error:
        logger.error(f"Cannot build the report: {error}")
        sys.exit(1)

    output_path = layout.report_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)

    logger.info(f"Report generated successfully at: {output_path}")
    logger.info(f"Runs selected on validation: {report['metadata']['selected_runs']}")


if __name__ == "__main__":
    main()
