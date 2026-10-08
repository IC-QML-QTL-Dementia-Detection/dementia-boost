"""Unit tests for the results layout: where each run's artifacts live.

Regression coverage
-------------------
- File names carried identity (`baseline_qtl_seed_3.pt`), so two configurations
  of one paradigm and seed collided, and the "skip if exists" check skipped a run
  whose configuration had changed.
- Nothing detected two different specs ending up under one ID.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dementia_boost.core.identity import Paradigm, RunSpec, config_id, label
from dementia_boost.core.layout import ConfigCollisionError, ResultsLayout


def _qtl(**overrides) -> RunSpec:
    """The PennyLane QTL spec used as the worked example (6 qubits, 4 layers)."""
    fields = {
        "paradigm": Paradigm.PL_QTL,
        "lr": 1e-4,
        "lr_step_size": 10,
        "lr_gamma": 0.75,
        "epochs": 100,
        "batch_size": 64,
        "split_id": "5192b1b7c0d3",
        "seed": 3,
        "ansatz": "paper",
        "n_qubits": 6,
        "n_layers": 4,
        "gradient": "adjoint",
        "backbone_id": "c8375944e970",
    }
    return RunSpec(**{**fields, **overrides})


@pytest.fixture
def layout(tmp_path: Path) -> ResultsLayout:
    """A layout rooted in a temporary directory."""
    return ResultsLayout(str(tmp_path))


class TestPaths:
    """Validates that every artifact path comes from the spec."""

    def test_paths_follow_the_documented_layout(
        self, layout: ResultsLayout, tmp_path: Path
    ) -> None:
        """Checkpoint, history, config, metrics, and plots sit under
        `<kind>/<paradigm>/<config_id>/`."""
        spec = _qtl()
        base = "pl_qtl/9e93de0f8105"
        assert layout.checkpoint_path(spec) == str(
            tmp_path / "checkpoints" / base / "seed_3.pt"
        )
        assert layout.history_path(spec) == str(
            tmp_path / "histories" / base / "seed_3.json"
        )
        assert layout.config_path(spec) == str(
            tmp_path / "histories" / base / "config.json"
        )
        assert layout.metrics_path("pl_qtl", "9e93de0f8105", "val") == str(
            tmp_path / "metrics" / base / "val_results.json"
        )
        assert layout.metrics_path("pl_qtl", "9e93de0f8105", "test") == str(
            tmp_path / "metrics" / base / "test_results.json"
        )
        assert layout.plots_dir("pl_qtl", "9e93de0f8105") == str(
            tmp_path / "plots" / base
        )

    def test_seeds_of_one_configuration_share_a_directory(
        self, layout: ResultsLayout
    ) -> None:
        """Different seeds of one configuration differ only in the file name."""
        first, second = _qtl(seed=3), _qtl(seed=4)
        assert (
            Path(layout.checkpoint_path(first)).parent
            == Path(layout.checkpoint_path(second)).parent
        )
        assert layout.checkpoint_path(first) != layout.checkpoint_path(second)

    def test_a_changed_configuration_gets_its_own_directory(
        self, layout: ResultsLayout
    ) -> None:
        """The same paradigm and seed with other layers does not collide."""
        other = dataclasses.replace(_qtl(), n_layers=3)
        assert layout.checkpoint_path(other) != layout.checkpoint_path(_qtl())
        assert (
            Path(layout.checkpoint_path(other)).parent
            != Path(layout.checkpoint_path(_qtl())).parent
        )

    def test_paradigm_may_be_given_as_a_string_or_enum(
        self, layout: ResultsLayout
    ) -> None:
        """Paths do not depend on how the paradigm is spelled."""
        assert layout.metrics_path(
            Paradigm.PL_QTL, "abc", "val"
        ) == layout.metrics_path("pl_qtl", "abc", "val")

    def test_unknown_cohort_for_metrics_raises(self, layout: ResultsLayout) -> None:
        """Only validation and test have a results file."""
        with pytest.raises(ValueError, match="cohort"):
            layout.metrics_path("pl_qtl", "abc", "train")


class TestHistoryFiles:
    """Validates listing of history files by paradigm and configuration."""

    def _touch(self, path: str) -> None:
        """Creates an empty file, with its directories."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    def test_lists_only_history_files(self, layout: ResultsLayout) -> None:
        """The config file and temporary files are not histories."""
        spec = _qtl()
        self._touch(layout.history_path(spec))
        self._touch(layout.config_path(spec))
        self._touch(layout.history_path(spec) + ".tmp")

        assert layout.history_files() == [layout.history_path(spec)]

    def test_filters_by_paradigm_and_configuration(self, layout: ResultsLayout) -> None:
        """A paradigm or a configuration ID narrows the listing."""
        first = _qtl()
        other_layers = dataclasses.replace(_qtl(), n_layers=3)
        baseline = _qtl(
            paradigm=Paradigm.BASELINE,
            ansatz=None,
            n_qubits=None,
            n_layers=None,
            gradient=None,
            backbone_id=None,
        )
        for spec in (first, other_layers, baseline):
            self._touch(layout.history_path(spec))

        assert len(layout.history_files()) == 3
        assert len(layout.history_files("pl_qtl")) == 2
        assert layout.history_files("pl_qtl", config_id(first)) == [
            layout.history_path(first)
        ]

    def test_nothing_on_disk_gives_an_empty_list(self, layout: ResultsLayout) -> None:
        """A layout without histories lists nothing."""
        assert layout.history_files() == []


def test_report_path_is_under_metrics(layout: ResultsLayout, tmp_path: Path) -> None:
    """The comparative report sits next to the per-configuration metrics."""
    assert layout.report_path() == str(tmp_path / "metrics" / "comparative_report.json")


class TestConfigsWithMetrics:
    """Validates listing of configurations that have a results file."""

    def test_lists_configurations_with_results_for_a_cohort(
        self, layout: ResultsLayout
    ) -> None:
        """Only configurations with the requested cohort's file are listed."""
        for configuration, cohort in (("bbb", "val"), ("aaa", "val"), ("ccc", "test")):
            path = Path(layout.metrics_path("baseline", configuration, cohort))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

        assert layout.configs_with_metrics("baseline", "val") == ["aaa", "bbb"]
        assert layout.configs_with_metrics("baseline", "test") == ["ccc"]
        assert layout.configs_with_metrics("ctl", "val") == []


class TestIsDone:
    """Validates the "skip if the checkpoint exists" check."""

    def _touch(self, path: str) -> None:
        """Creates an empty file, with its directories."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    def test_not_done_until_the_checkpoint_exists(self, layout: ResultsLayout) -> None:
        """A run is done once its checkpoint is on disk."""
        spec = _qtl()
        assert not layout.is_done(spec)
        self._touch(layout.checkpoint_path(spec))
        assert layout.is_done(spec)

    def test_a_rerun_of_the_same_spec_is_skipped(self, layout: ResultsLayout) -> None:
        """An identical spec is recognised as done."""
        self._touch(layout.checkpoint_path(_qtl()))
        assert layout.is_done(_qtl())

    def test_a_spec_differing_only_in_layers_is_not_skipped(
        self, layout: ResultsLayout
    ) -> None:
        """A changed configuration is not mistaken for a finished run."""
        self._touch(layout.checkpoint_path(_qtl()))
        assert not layout.is_done(dataclasses.replace(_qtl(), n_layers=3))

    def test_another_seed_is_not_done(self, layout: ResultsLayout) -> None:
        """A finished seed does not mark the other seeds as done."""
        self._touch(layout.checkpoint_path(_qtl(seed=3)))
        assert not layout.is_done(_qtl(seed=4))


class TestWriteConfig:
    """Validates `config.json` and the collision check."""

    def test_writes_the_spec_without_the_seed_and_the_label(
        self, layout: ResultsLayout
    ) -> None:
        """The config file names the configuration, not one seed."""
        spec = _qtl()
        layout.write_config(spec)

        payload = json.loads(Path(layout.config_path(spec)).read_text())

        expected = spec.to_dict()
        del expected["seed"]
        assert payload == {
            "config_id": config_id(spec),
            "label": label(spec, include_seed=False),
            "spec": expected,
        }

    def test_other_seeds_of_the_configuration_can_be_written_again(
        self, layout: ResultsLayout
    ) -> None:
        """Writing the config for a second seed leaves the file unchanged."""
        layout.write_config(_qtl(seed=3))
        before = Path(layout.config_path(_qtl())).read_bytes()

        layout.write_config(_qtl(seed=4))

        assert Path(layout.config_path(_qtl())).read_bytes() == before

    def test_two_specs_under_one_id_are_refused(
        self, layout: ResultsLayout, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hash collision, simulated by forcing equal IDs, is detected instead
        of silently merging two configurations."""
        monkeypatch.setattr("dementia_boost.core.layout.config_id", lambda spec: "same")
        layout.write_config(_qtl())

        with pytest.raises(ConfigCollisionError, match="same"):
            layout.write_config(dataclasses.replace(_qtl(), n_layers=3))

    def test_no_temporary_file_is_left_behind(self, layout: ResultsLayout) -> None:
        """The config is written atomically."""
        layout.write_config(_qtl())
        leftovers = list(Path(layout.config_path(_qtl())).parent.glob("*.tmp"))
        assert leftovers == []
