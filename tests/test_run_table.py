"""Tests for the human lookup table of runs and configurations.

Regression coverage
-------------------
- A hash ID says nothing to a person; without a lookup there was no quick way to
  find which configuration and seed an ID belongs to.
- Looking an ID up must read the stored specs, not guess from file names.
"""

import csv
from pathlib import Path

import pytest
from conftest import build_spec

from dementia_boost.core.identity import RunSpec, config_id, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.telemetry.run_listing import load_runs
from dementia_boost.telemetry.run_table import (
    configuration_rows,
    find_matches,
    format_table,
    run_rows,
    write_csv,
)


def _write(layout: ResultsLayout, spec: RunSpec, finished: bool = True) -> None:
    """Writes a history (and optionally a checkpoint) for a spec."""
    history_path = Path(layout.history_path(spec))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    MetricsAnalyzer.save_history(
        TrainingHistory(
            spec=spec, epochs=[EpochRecord(1, 0.7, 0.5, 0.7, 0.5, spec.lr, 1.0)]
        ),
        str(history_path),
    )
    layout.write_config(spec)
    if finished:
        checkpoint = Path(layout.checkpoint_path(spec))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"")


@pytest.fixture
def populated(tmp_path: Path):
    """A layout with two PL QTL configurations and one baseline."""
    layout = ResultsLayout(str(tmp_path))
    specs = [
        build_spec("pl_qtl", seed=3),
        build_spec("pl_qtl", seed=4),
        build_spec("pl_qtl", seed=3, n_layers=2),
        build_spec("baseline", seed=1),
    ]
    for spec in specs:
        _write(layout, spec)
    return layout, specs, load_runs(layout)


class TestRows:
    """Validates the table rows."""

    def test_one_row_per_configuration_with_its_seeds(self, populated) -> None:
        """A configuration row names the label and lists the seeds."""
        layout, specs, runs = populated

        rows = configuration_rows(runs)

        assert len(rows) == 3
        paper = next(r for r in rows if r["config_id"] == config_id(specs[0]))
        assert paper["paradigm"] == "pl_qtl"
        assert paper["label"] == "pl_qtl | paper | 2q x 1L | lr 0.001"
        assert paper["seeds"] == "3,4"
        assert paper["n_runs"] == 2
        assert paper["split_id"] == "5192b1b7c0d3"

    def test_one_row_per_run_with_ids_and_paths(self, populated) -> None:
        """A run row connects the run ID to its seed, configuration, and files."""
        layout, specs, runs = populated

        rows = run_rows(runs, layout)

        assert len(rows) == 4
        row = next(r for r in rows if r["run_id"] == run_id(specs[1]))
        assert row["seed"] == 4
        assert row["config_id"] == config_id(specs[1])
        assert row["finished"] == "yes"
        assert row["checkpoint"] == layout.checkpoint_path(specs[1])
        assert row["history"] == layout.history_path(specs[1])

    def test_unfinished_runs_are_marked(self, tmp_path: Path) -> None:
        """A run with a history but no checkpoint is shown as not finished."""
        layout = ResultsLayout(str(tmp_path))
        _write(layout, build_spec("baseline", seed=1), finished=False)

        assert run_rows(load_runs(layout), layout)[0]["finished"] == "no"


class TestFindMatches:
    """Validates looking a hash up by prefix."""

    def test_a_run_id_prefix_finds_the_run_and_its_full_spec(self, populated) -> None:
        """A run ID leads to its seed, its full spec, and its files."""
        layout, specs, runs = populated
        target = specs[1]

        matches = find_matches(runs, layout, run_id(target)[:6])

        assert len(matches) == 1
        match = matches[0]
        assert match["kind"] == "run"
        assert match["spec"] == target.to_dict()
        assert match["files"]["checkpoint"] == layout.checkpoint_path(target)

    def test_a_config_id_prefix_finds_the_configuration_and_its_runs(
        self, populated
    ) -> None:
        """A configuration ID leads to its spec without the seed and its runs."""
        layout, specs, runs = populated

        matches = find_matches(runs, layout, config_id(specs[0])[:6])

        assert len(matches) == 1
        match = matches[0]
        assert match["kind"] == "configuration"
        assert "seed" not in match["spec"]
        assert match["runs"] == [
            {"seed": 3, "run_id": run_id(specs[0])},
            {"seed": 4, "run_id": run_id(specs[1])},
        ]
        assert match["files"]["config"] == layout.config_path(specs[0])

    def test_an_unknown_prefix_finds_nothing(self, populated) -> None:
        """An ID that is not on disk returns no matches."""
        layout, _, runs = populated
        assert find_matches(runs, layout, "zzzzzz") == []

    def test_a_too_short_prefix_is_refused(self, populated) -> None:
        """A one or two character prefix would match almost anything."""
        layout, _, runs = populated
        with pytest.raises(ValueError, match="at least"):
            find_matches(runs, layout, "a")


class TestOutput:
    """Validates the text table and the CSV export."""

    def test_table_has_a_header_and_aligned_columns(self) -> None:
        """Columns are padded to the widest cell."""
        rows = [{"a": "x", "b": 1}, {"a": "longer", "b": 22}]

        lines = format_table(rows, ["a", "b"]).splitlines()

        assert lines[0].split() == ["a", "b"]
        assert set(lines[1].replace(" ", "")) == {"-"}
        offset = lines[0].index("b")
        assert lines[2][offset:].strip() == "1"
        assert lines[3][offset:].strip() == "22"
        assert lines[3].startswith("longer")

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        """The CSV holds the same columns and values as the rows."""
        rows = [{"a": "x", "b": 1}, {"a": "y", "b": 2}]
        path = tmp_path / "runs.csv"

        write_csv(rows, ["a", "b"], str(path))

        with open(path, newline="") as handle:
            assert list(csv.DictReader(handle)) == [
                {"a": "x", "b": "1"},
                {"a": "y", "b": "2"},
            ]

    def test_empty_table_says_so(self) -> None:
        """No rows gives a readable message instead of a bare header."""
        assert "No runs" in format_table([], ["a"])
