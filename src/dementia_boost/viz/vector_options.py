"""Command-line options shared by the visualization scripts that write vector figures.

`--svg` and `--pdf` make a script write each cross-configuration figure in that
format next to its PNG, with the same name, so a poster or paper can embed it
without losing resolution.
"""

import argparse

from dementia_boost.viz.visualizer import MetricsVisualizer


def add_vector_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the `--svg` and `--pdf` flags to a parser.

    Args:
        parser: The parser of a visualization script.
    """
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also write each figure as an SVG file next to the PNG.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write each figure as a PDF file next to the PNG.",
    )


def vector_formats(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Lists the vector formats the parsed flags ask for.

    Args:
        arguments: The parsed arguments of a script that called
            `add_vector_arguments`.

    Returns:
        The requested formats, in the order of `MetricsVisualizer.VECTOR_FORMATS`.
    """
    return tuple(
        extension
        for extension in MetricsVisualizer.VECTOR_FORMATS
        if getattr(arguments, extension)
    )
