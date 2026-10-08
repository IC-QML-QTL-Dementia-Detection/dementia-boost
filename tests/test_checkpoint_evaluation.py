"""Unit tests for evaluating saved checkpoints on the validation and test cohorts."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from conftest import build_spec
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.core.identity import RunSpec, config_id, run_id
from dementia_boost.core.layout import ResultsLayout, config_payload
from dementia_boost.telemetry.metrics import (
    EpochRecord,
    MetricsAnalyzer,
    TrainingHistory,
)
from dementia_boost.training.checkpoint_evaluation import (
    evaluate_checkpoints,
    evaluate_paradigm,
    save_configuration_results,
)
from dementia_boost.training.evaluator import ModelEvaluator


def _model() -> nn.Module:
    """A tiny logit model over 1x2x2 inputs."""
    return nn.Sequential(nn.Flatten(), nn.Linear(4, 1))


def _loader(seed: int) -> DataLoader:
    """A small two-class loader whose labels follow the first feature."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(12, 1, 2, 2, generator=generator)
    labels = (images[:, 0, 0, 0] > 0).long()
    labels[:2] = torch.tensor([0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def _checkpoints(tmp_path: Path, n: int = 3) -> list[tuple[str, str]]:
    """Saves n differently initialised checkpoints and returns (run_id, path)."""
    saved = []
    for i in range(n):
        torch.manual_seed(i)
        path = tmp_path / f"run_{i}.pt"
        torch.save(_model().state_dict(), path)
        saved.append((f"run_{i}", str(path)))
    return saved


def _train_run(layout: ResultsLayout, spec: RunSpec, seed_for_weights: int) -> None:
    """Writes a history and a checkpoint for a spec, as the trainer would."""
    history_path = Path(layout.history_path(spec))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    MetricsAnalyzer.save_history(
        TrainingHistory(
            spec=spec,
            epochs=[EpochRecord(1, 0.7, 0.5, 0.7, 0.5, spec.lr, 1.0)],
            extras={},
        ),
        str(history_path),
    )
    checkpoint_path = Path(layout.checkpoint_path(spec))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed_for_weights)
    torch.save(_model().state_dict(), checkpoint_path)


def test_every_checkpoint_is_evaluated_on_every_cohort(tmp_path: Path) -> None:
    """Each cohort gets one result per checkpoint, in checkpoint order."""
    evaluator = ModelEvaluator(_model(), torch.device("cpu"))
    loaders = {"val": _loader(1), "test": _loader(2)}

    results = evaluate_checkpoints(evaluator, _checkpoints(tmp_path), loaders)

    assert set(results) == {"val", "test"}
    for cohort in ("val", "test"):
        assert [r.run_id for r in results[cohort]] == ["run_0", "run_1", "run_2"]


def test_metrics_match_direct_inference_per_cohort(tmp_path: Path) -> None:
    """Values for a cohort come from that cohort's loader, not the other one."""
    evaluator = ModelEvaluator(_model(), torch.device("cpu"))
    loaders = {"val": _loader(1), "test": _loader(2)}
    checkpoints = _checkpoints(tmp_path, n=1)

    results = evaluate_checkpoints(evaluator, checkpoints, loaders)

    evaluator.load_weights(checkpoints[0][1])
    for cohort, loader in loaders.items():
        y_true, y_prob = evaluator.predict(loader)
        expected = MetricsAnalyzer.calculate_metrics("run_0", y_true, y_prob)
        assert results[cohort][0].auc == expected.auc
        assert np.allclose(results[cohort][0].y_prob, expected.y_prob)
    assert results["val"][0].y_prob != results["test"][0].y_prob


def test_results_are_saved_per_cohort_with_the_configuration(
    tmp_path: Path,
) -> None:
    """Validation and test metrics go to separate files that name their cohort
    and carry the configuration they were computed for."""
    layout = ResultsLayout(str(tmp_path / "results"))
    spec = build_spec("baseline", seed=1)
    evaluator = ModelEvaluator(_model(), torch.device("cpu"))
    loaders = {"val": _loader(1), "test": _loader(2)}
    results = evaluate_checkpoints(evaluator, _checkpoints(tmp_path), loaders)

    paths = save_configuration_results(layout, spec, results)

    configuration = config_id(spec)
    assert paths["val"] == layout.metrics_path("baseline", configuration, "val")
    assert paths["test"] == layout.metrics_path("baseline", configuration, "test")
    for cohort, path in paths.items():
        payload = json.loads(Path(path).read_text())
        assert payload["cohort"] == cohort
        assert payload["configuration"] == config_payload(spec)
        assert len(payload["individual_runs"]) == 3
        assert set(payload["aggregated_statistics"]) >= {"auc", "accuracy"}


class TestEvaluateParadigm:
    """Validates evaluating every configuration of a paradigm."""

    def _evaluate(self, layout: ResultsLayout, calls: list[RunSpec]):
        """Runs `evaluate_paradigm` with a recording model builder."""

        def build_model(spec: RunSpec) -> nn.Module:
            calls.append(spec)
            return _model()

        return evaluate_paradigm(
            layout,
            "baseline",
            build_model,
            {"val": _loader(1), "test": _loader(2)},
            torch.device("cpu"),
            logging.getLogger("test_evaluate_paradigm"),
        )

    def test_each_configuration_gets_its_own_results_with_its_own_runs(
        self, tmp_path: Path
    ) -> None:
        """Seeds of one configuration are evaluated together, and another
        configuration is evaluated and saved separately."""
        layout = ResultsLayout(str(tmp_path))
        specs = [
            build_spec("baseline", seed=1),
            build_spec("baseline", seed=2),
            build_spec("baseline", seed=1, lr=5e-4),
        ]
        for index, spec in enumerate(specs):
            _train_run(layout, spec, index)

        outcome = self._evaluate(layout, [])

        assert set(outcome) == {config_id(specs[0]), config_id(specs[2])}
        first = json.loads(
            Path(
                layout.metrics_path("baseline", config_id(specs[0]), "val")
            ).read_text()
        )
        assert [r["run_id"] for r in first["individual_runs"]] == [
            run_id(specs[0]),
            run_id(specs[1]),
        ]
        other = json.loads(
            Path(
                layout.metrics_path("baseline", config_id(specs[2]), "test")
            ).read_text()
        )
        assert [r["run_id"] for r in other["individual_runs"]] == [run_id(specs[2])]

    def test_the_model_is_built_from_the_spec_of_each_configuration(
        self, tmp_path: Path
    ) -> None:
        """The architecture comes from the configuration, once per configuration."""
        layout = ResultsLayout(str(tmp_path))
        specs = [
            build_spec("baseline", seed=1),
            build_spec("baseline", seed=1, lr=5e-4),
        ]
        for index, spec in enumerate(specs):
            _train_run(layout, spec, index)
        calls: list[RunSpec] = []

        self._evaluate(layout, calls)

        assert sorted(config_id(spec) for spec in calls) == sorted(
            config_id(spec) for spec in specs
        )

    def test_runs_without_a_checkpoint_are_left_out(self, tmp_path: Path) -> None:
        """A run that did not finish has a history but no checkpoint, and is
        not evaluated."""
        layout = ResultsLayout(str(tmp_path))
        finished, unfinished = (
            build_spec("baseline", seed=1),
            build_spec("baseline", seed=2),
        )
        _train_run(layout, finished, 0)
        _train_run(layout, unfinished, 1)
        Path(layout.checkpoint_path(unfinished)).unlink()

        self._evaluate(layout, [])

        payload = json.loads(
            Path(
                layout.metrics_path("baseline", config_id(finished), "val")
            ).read_text()
        )
        assert [r["run_id"] for r in payload["individual_runs"]] == [run_id(finished)]

    def test_nothing_to_evaluate_returns_nothing(self, tmp_path: Path) -> None:
        """A layout without histories evaluates nothing and writes nothing."""
        layout = ResultsLayout(str(tmp_path))

        assert self._evaluate(layout, []) == {}
        assert list(Path(tmp_path).rglob("*.json")) == []
