"""Unit tests for evaluating saved checkpoints on the validation and test cohorts."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.telemetry.metrics import MetricsAnalyzer
from dementia_boost.training.checkpoint_evaluation import (
    cohort_results_path,
    evaluate_checkpoints,
    save_cohort_results,
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


def test_results_are_saved_per_cohort_with_the_cohort_marker(tmp_path: Path) -> None:
    """Validation and test metrics go to separate files that name their cohort."""
    evaluator = ModelEvaluator(_model(), torch.device("cpu"))
    loaders = {"val": _loader(1), "test": _loader(2)}
    results = evaluate_checkpoints(evaluator, _checkpoints(tmp_path), loaders)

    paths = save_cohort_results(results, str(tmp_path / "metrics"), "base")

    assert paths["test"] == str(tmp_path / "metrics" / "base_results.json")
    assert paths["val"] == str(tmp_path / "metrics" / "base_val_results.json")
    for cohort, path in paths.items():
        payload = json.loads(Path(path).read_text())
        assert payload["cohort"] == cohort
        assert len(payload["individual_runs"]) == 3
        assert set(payload["aggregated_statistics"]) >= {"auc", "accuracy"}


def test_cohort_results_path_convention() -> None:
    """The test file keeps the plain name; validation gets a `_val` suffix."""
    assert cohort_results_path("m", "tl", "test") == "m/tl_results.json"
    assert cohort_results_path("m", "tl", "val") == "m/tl_val_results.json"
