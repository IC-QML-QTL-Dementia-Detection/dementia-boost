"""Unit tests for listing runs from their history files.

Regression coverage
-------------------
- Evaluation scripts recovered the seed and configuration of a run by stripping
  prefixes off file names (`baseline_qtl_seed_3.pt`), which breaks as soon as a
  name differs in more than the seed.
- Nothing noticed a history file that sat in the wrong configuration directory.
"""

import shutil
from pathlib import Path

import pytest
from conftest import build_spec

from dementia_boost.core.identity import RunSpec, config_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.telemetry.run_listing import group_by_config, load_runs


def _write(layout: ResultsLayout, spec: RunSpec) -> str:
    """Writes a one-epoch history for a spec at its layout path."""
    path = Path(layout.history_path(spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    history = TrainingHistory(
        spec=spec,
        epochs=[EpochRecord(1, 0.7, 0.5, 0.7, 0.5, spec.lr, 1.0)],
        extras={},
    )
    MetricsAnalyzer.save_history(history, str(path))
    return str(path)


@pytest.fixture
def layout(tmp_path: Path) -> ResultsLayout:
    """A layout rooted in a temporary directory."""
    return ResultsLayout(str(tmp_path))


class TestLoadRuns:
    """Validates discovery of runs from histories."""

    def test_returns_every_run_sorted_by_paradigm_config_and_seed(
        self, layout: ResultsLayout
    ) -> None:
        """All histories are found, in a stable order."""
        specs = [
            build_spec("pl_qtl", seed=4),
            build_spec("pl_qtl", seed=3),
            build_spec("pl_qtl", seed=3, n_layers=2),
            build_spec("baseline", seed=2),
        ]
        for spec in specs:
            _write(layout, spec)

        runs = load_runs(layout)

        assert len(runs) == 4
        keys = [(r.spec.paradigm.value, config_id(r.spec), r.spec.seed) for r in runs]
        assert keys == sorted(keys)

    def test_filters_by_paradigm_and_configuration(self, layout: ResultsLayout) -> None:
        """A paradigm or a configuration narrows the result."""
        target = build_spec("pl_qtl", seed=3)
        for spec in (
            target,
            build_spec("pl_qtl", seed=4),
            build_spec("pl_qtl", n_layers=2),
            build_spec("baseline"),
        ):
            _write(layout, spec)

        assert {r.spec.paradigm.value for r in load_runs(layout, "pl_qtl")} == {
            "pl_qtl"
        }
        by_config = load_runs(layout, "pl_qtl", config_id(target))
        assert [r.spec.seed for r in by_config] == [3, 4]
        assert {config_id(r.spec) for r in by_config} == {config_id(target)}

    def test_ignores_the_config_file_and_temporary_files(
        self, layout: ResultsLayout
    ) -> None:
        """Only history files count."""
        spec = build_spec("pl_qtl", seed=3)
        history_path = _write(layout, spec)
        layout.write_config(spec)
        Path(history_path + ".tmp").write_text("{}")

        assert len(load_runs(layout)) == 1

    def test_empty_results_give_no_runs(self, layout: ResultsLayout) -> None:
        """A layout with nothing in it lists nothing."""
        assert load_runs(layout) == []

    def test_a_history_in_the_wrong_directory_is_refused(
        self, layout: ResultsLayout
    ) -> None:
        """A file moved into another configuration's directory disagrees with
        the path its own spec gives, so it is not trusted."""
        spec = build_spec("pl_qtl", seed=3)
        other = build_spec("pl_qtl", seed=3, n_layers=2)
        source = _write(layout, spec)
        destination = Path(layout.history_path(other))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)

        with pytest.raises(ValueError, match="does not belong"):
            load_runs(layout)

    def test_a_renamed_seed_file_is_refused(self, layout: ResultsLayout) -> None:
        """The seed comes from the spec, never from the file name."""
        spec = build_spec("pl_qtl", seed=3)
        source = Path(_write(layout, spec))
        source.rename(source.with_name("seed_9.json"))

        with pytest.raises(ValueError, match="does not belong"):
            load_runs(layout)


class TestGroupByConfig:
    """Validates grouping of runs into configurations."""

    def test_groups_seeds_of_one_configuration(self, layout: ResultsLayout) -> None:
        """Seeds share a configuration, a changed layer count is another one."""
        for spec in (
            build_spec("pl_qtl", seed=4),
            build_spec("pl_qtl", seed=3),
            build_spec("pl_qtl", seed=3, n_layers=2),
        ):
            _write(layout, spec)

        groups = group_by_config(load_runs(layout))

        assert len(groups) == 2
        seeds_by_size = sorted(
            [[r.spec.seed for r in runs] for runs in groups.values()], key=len
        )
        assert seeds_by_size == [[3], [3, 4]]

    def test_keys_are_configuration_ids(self, layout: ResultsLayout) -> None:
        """Groups are keyed by `config_id`."""
        spec = build_spec("pl_qtl")
        _write(layout, spec)

        assert list(group_by_config(load_runs(layout))) == [config_id(spec)]
