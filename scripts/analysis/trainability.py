"""Trainability of every registered circuit at initialisation, without training.

Simulates each circuit exactly at random initial weights and random inputs and
reports its input spread, input sensitivity, and the scale and variance of its
weight gradients for two costs (the mean of all `<Z_i>` and `<Z_0>` alone),
across qubit counts (see `dementia_boost.analysis.trainability`). Writes one
JSON file and prints a Markdown table.

    uv run scripts/analysis/trainability.py
    uv run scripts/analysis/trainability.py --qubits 4 6 --draws 10
"""

import argparse
import json
import os

from dementia_boost.analysis.trainability import (
    DEFAULT_INPUTS,
    DEFAULT_LAYERS,
    DEFAULT_QUBITS,
    DEFAULT_WEIGHT_DRAWS,
    default_choices,
    diagnose,
    markdown_table,
)
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parses the command-line options of the diagnostic.

    Returns:
        The namespace with `qubits`, `layers`, `draws`, `inputs`, `seed`, and
        `output`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUBITS),
        help="Qubit counts to measure each circuit at.",
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_WEIGHT_DRAWS,
        help="Random weight initialisations per circuit and size.",
    )
    parser.add_argument(
        "--inputs",
        type=int,
        default=DEFAULT_INPUTS,
        help="Random inputs per weight draw for the input spread.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=os.path.join(ResultsLayout().root, "diagnostics", "trainability.json"),
        help="Path of the JSON file with every result.",
    )
    return parser.parse_args()


def main() -> None:
    """Measures every default circuit at every qubit count and saves the results."""
    arguments = parse_arguments()
    logger = setup_logger("trainability")

    results = []
    for choice in default_choices():
        for n_qubits in arguments.qubits:
            logger.info(
                f"Measuring {choice.embedding} embedding, {choice.block} block, "
                f"re-uploading {choice.reupload}, {n_qubits} qubits"
            )
            results.append(
                diagnose(
                    choice,
                    n_qubits=n_qubits,
                    n_layers=arguments.layers,
                    n_weight_draws=arguments.draws,
                    n_inputs=arguments.inputs,
                    seed=arguments.seed,
                )
            )

    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    with open(arguments.output, "w") as file:
        json.dump(
            {
                "settings": {
                    "qubits": arguments.qubits,
                    "layers": arguments.layers,
                    "draws": arguments.draws,
                    "inputs": arguments.inputs,
                    "seed": arguments.seed,
                },
                "results": [result.to_dict() for result in results],
            },
            file,
            indent=2,
        )
    logger.info(f"Saved {len(results)} result(s) to {arguments.output}")
    print(markdown_table(results))


if __name__ == "__main__":
    main()
