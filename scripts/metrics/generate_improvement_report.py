"""Comparative improvement report generator across paradigms on NIfTI data.

This script aggregates evaluation results across the Classical Baseline, Classical
Transfer Learning (CTL), PennyLane Quantum Transfer Learning (QTL), and Qiskit
Quantum Transfer Learning (Qiskit QTL) runs on NIfTI data. It reports the mean
and standard deviation of every metric across seeds on the test cohort, shows
the test metrics of the run selected on the validation cohort, calculates the
relative percentage change between the means, and exports a comparative JSON
report. No run is chosen on test metrics.
"""

import json
import os

from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.report import build_comparative_report
from dementia_boost.training.checkpoint_evaluation import cohort_results_path

RESULTS_DIR = "./data/results/metrics/nifti"
OUTPUT_PATH = f"{RESULTS_DIR}/comparative_report.json"
RESULT_STEMS = {
    "baseline": "baseline",
    "ctl": "tl",
    "qtl": "qtl",
    "qiskit_qtl": "qiskit_qtl",
}
REQUIRED_PARADIGMS = ("baseline", "ctl")


def main() -> None:
    """Generates the quantitative comparative improvement report for NIfTI."""
    logger = setup_logger("improvement_report_nifti")

    test_results: dict[str, dict] = {}
    val_results: dict[str, dict] = {}
    for paradigm, stem in RESULT_STEMS.items():
        paths = {
            "test": cohort_results_path(RESULTS_DIR, stem, "test"),
            "val": cohort_results_path(RESULTS_DIR, stem, "val"),
        }
        missing = [path for path in paths.values() if not os.path.exists(path)]
        if missing:
            if paradigm in REQUIRED_PARADIGMS:
                logger.error(f"Missing telemetry for {paradigm}: {missing}")
                return
            logger.info(f"Skipping {paradigm}: missing {missing}")
            continue
        with open(paths["test"]) as file:
            test_results[paradigm] = json.load(file)
        with open(paths["val"]) as file:
            val_results[paradigm] = json.load(file)

    report = build_comparative_report(test_results, val_results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    logger.info(f"Report generated successfully at: {OUTPUT_PATH}")
    logger.info(f"Runs selected on validation: {report['metadata']['selected_runs']}")


if __name__ == "__main__":
    main()
