"""Unit tests for BaselineTrainer lifecycle and checkpointing.

This module validates that BaselineTrainer supports both end-to-end full model
training on raw images and fast transfer learning on cached feature embeddings
with composite model checkpoint serialization for CTL and QTL heads, that it
takes its identity from a `RunSpec` and refuses a spec that does not describe
what actually runs, and that it records a per-epoch `TrainingHistory` carrying
the spec, persists it atomically every few epochs and on exit, and rejects
non-positive evaluation or save cadences.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from conftest import build_spec
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from dementia_boost.core.identity import Paradigm, RunSpec, config_id, run_id
from dementia_boost.core.layout import ResultsLayout
from dementia_boost.data.embedding_cache import FeatureCacheManager
from dementia_boost.models.builder import assemble_dementia_classifier
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import PennylaneQuantumClassifierHead
from dementia_boost.telemetry.logger import setup_logger
from dementia_boost.telemetry.metrics import MetricsAnalyzer, TrainingHistory
from dementia_boost.training.evaluator import ModelEvaluator
from dementia_boost.training.trainer import BaselineTrainer

NUM_MOCK_SAMPLES: int = 8
NUM_CHANNELS: int = 1
IMAGE_SIZE: int = 128
BATCH_SIZE: int = 4
FEATURE_DIM: int = 2304
LEARNING_RATE: float = 1e-3
STEP_SIZE: int = 5
GAMMA: float = 0.5
TEST_QTL_QUBITS: int = 2
TEST_QTL_LAYERS: int = 1
HISTORY_STEP_SIZE: int = 2
BATCHES_PER_EPOCH: int = NUM_MOCK_SAMPLES // BATCH_SIZE


def _spec(paradigm: Paradigm | str = Paradigm.BASELINE, **overrides) -> RunSpec:
    """Builds a spec that matches the optimizer, scheduler, and loaders below."""
    return build_spec(
        paradigm,
        **{
            "lr": LEARNING_RATE,
            "lr_step_size": STEP_SIZE,
            "lr_gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            **overrides,
        },
    )


def _cached_loaders(
    extractor: nn.Module, device: torch.device
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Builds a raw loader plus train and validation loaders of cached embeddings."""
    raw_images = torch.randn(NUM_MOCK_SAMPLES, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    raw_labels = torch.tensor([0.0, 1.0] * (NUM_MOCK_SAMPLES // 2))
    raw_loader = DataLoader(
        TensorDataset(raw_images, raw_labels), batch_size=BATCH_SIZE, shuffle=False
    )
    features, labels = FeatureCacheManager.extract_features(
        feature_extractor=extractor, data_loader=raw_loader, device=device
    )
    train_loader = FeatureCacheManager.create_cached_loader(
        features=features, labels=labels, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = FeatureCacheManager.create_cached_loader(
        features=features, labels=labels, batch_size=BATCH_SIZE, shuffle=False
    )
    return raw_loader, train_loader, val_loader


def test_baseline_trainer_with_cached_embeddings_and_save_model(
    tmp_path: Path,
) -> None:
    """Validates training a head on cached embeddings and saving assembled model."""
    device = torch.device("cpu")
    layout = ResultsLayout(str(tmp_path))
    spec = _spec(Paradigm.CTL)

    extractor = LeNetFeatureExtractor().to(device)
    for param in extractor.parameters():
        param.requires_grad = False
    raw_loader, train_loader, val_loader = _cached_loaders(extractor, device)

    head = ClassicalClassifierHead(in_features=FEATURE_DIM, use_sigmoid=False).to(
        device
    )
    head.apply(ClassicalClassifierHead.apply_glorot_init)
    full_model = assemble_dementia_classifier(
        feature_extractor=extractor, classifier_head=head
    )

    optimizer = optim.Adam(head.parameters(), lr=LEARNING_RATE)
    trainer = BaselineTrainer(
        model=head,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA),
        device=device,
        logger=setup_logger("test_trainer_ctl"),
        spec=spec,
        layout=layout,
        save_model=full_model,
    )

    trainer.train()

    saved_checkpoint_path = Path(layout.checkpoint_path(spec))
    assert saved_checkpoint_path.exists()

    state_dict = torch.load(saved_checkpoint_path, weights_only=True)
    assert any(k.startswith("feature_extractor.") for k in state_dict.keys())
    assert any(k.startswith("classifier_head.") for k in state_dict.keys())

    evaluator_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )
    evaluator = ModelEvaluator(model=evaluator_model, device=device)
    evaluator.load_weights(str(saved_checkpoint_path))

    y_true, y_prob = evaluator.predict(raw_loader)
    assert y_true.shape == (NUM_MOCK_SAMPLES,)
    assert y_prob.shape == (NUM_MOCK_SAMPLES,)
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()


def test_baseline_trainer_with_quantum_head_and_cached_embeddings(
    tmp_path: Path,
) -> None:
    """Validates training a quantum head on cached embeddings and saving model."""
    device = torch.device("cpu")
    layout = ResultsLayout(str(tmp_path))
    spec = _spec(
        Paradigm.PL_QTL,
        n_qubits=TEST_QTL_QUBITS,
        n_layers=TEST_QTL_LAYERS,
        epochs=1,
    )

    extractor = LeNetFeatureExtractor().to(device)
    for param in extractor.parameters():
        param.requires_grad = False
    raw_loader, train_loader, val_loader = _cached_loaders(extractor, device)

    qtl_head = PennylaneQuantumClassifierHead(
        in_features=FEATURE_DIM,
        n_qubits=TEST_QTL_QUBITS,
        n_layers=TEST_QTL_LAYERS,
    ).to(device)
    qtl_head.apply(PennylaneQuantumClassifierHead.apply_glorot_init)
    full_model = assemble_dementia_classifier(
        feature_extractor=extractor, classifier_head=qtl_head
    )

    optimizer = optim.Adam(qtl_head.parameters(), lr=LEARNING_RATE)
    trainer = BaselineTrainer(
        model=qtl_head,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA),
        device=device,
        logger=setup_logger("test_trainer_qtl"),
        spec=spec,
        layout=layout,
        save_model=full_model,
    )

    trainer.train()

    saved_checkpoint_path = Path(layout.checkpoint_path(spec))
    assert saved_checkpoint_path.exists()

    state_dict = torch.load(saved_checkpoint_path, weights_only=True)
    assert any(k.startswith("feature_extractor.") for k in state_dict.keys())
    assert any(k.startswith("classifier_head.") for k in state_dict.keys())

    evaluator_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=PennylaneQuantumClassifierHead(
            in_features=FEATURE_DIM,
            n_qubits=TEST_QTL_QUBITS,
            n_layers=TEST_QTL_LAYERS,
        ),
    )
    evaluator = ModelEvaluator(model=evaluator_model, device=device)
    evaluator.load_weights(str(saved_checkpoint_path))

    y_true, y_prob = evaluator.predict(raw_loader)
    assert y_true.shape == (NUM_MOCK_SAMPLES,)
    assert y_prob.shape == (NUM_MOCK_SAMPLES,)
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()


class _FailingCriterion(nn.BCEWithLogitsLoss):
    """BCE loss that raises after a fixed number of calls to simulate a crash."""

    def __init__(self, max_calls: int) -> None:
        super().__init__()
        self.max_calls = max_calls
        self.calls = 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("simulated crash")
        return super().forward(input, target)


def _build_history_trainer(
    tmp_path: Path,
    criterion: nn.Module,
    eval_every: int,
    history_save_every: int = 10,
    epochs: int = 4,
    spec: RunSpec | None = None,
    extras: dict | None = None,
) -> tuple[BaselineTrainer, Path]:
    """Builds a tiny linear-model trainer that records its history to disk."""
    features = torch.randn(NUM_MOCK_SAMPLES, 4)
    labels = torch.tensor([0.0, 1.0] * (NUM_MOCK_SAMPLES // 2))
    loader = DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE)
    model = nn.Linear(4, 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    layout = ResultsLayout(str(tmp_path))
    spec = spec or _spec(lr_step_size=HISTORY_STEP_SIZE, epochs=epochs)

    trainer = BaselineTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=HISTORY_STEP_SIZE, gamma=GAMMA),
        device=torch.device("cpu"),
        logger=setup_logger("test_trainer_history"),
        spec=spec,
        layout=layout,
        eval_every=eval_every,
        history_save_every=history_save_every,
        extras=extras,
    )
    return trainer, Path(layout.history_path(spec))


def test_train_returns_history_matching_schedule_and_disk(tmp_path: Path) -> None:
    """Validates epoch count, finite losses, the StepLR learning rate used in
    each epoch, validation only on `eval_every` boundaries, and that the JSON
    on disk equals the returned history."""
    trainer, history_path = _build_history_trainer(
        tmp_path, nn.BCEWithLogitsLoss(), eval_every=2
    )

    history = trainer.train()

    assert [r.epoch for r in history.epochs] == [1, 2, 3, 4]
    assert all(math.isfinite(r.train_loss) for r in history.epochs)
    assert [r.lr for r in history.epochs] == pytest.approx(
        [LEARNING_RATE, LEARNING_RATE, LEARNING_RATE * GAMMA, LEARNING_RATE * GAMMA]
    )
    assert [r.val_loss is not None for r in history.epochs] == [
        False,
        True,
        False,
        True,
    ]
    assert history.spec.lr_step_size == HISTORY_STEP_SIZE
    assert MetricsAnalyzer.load_history(str(history_path)) == history


def test_history_carries_the_spec_and_the_non_identity_extras(tmp_path: Path) -> None:
    """Validates that the history holds the full spec (so runs can be listed
    without parsing names) and records the cadence and caller extras separately."""
    trainer, _ = _build_history_trainer(
        tmp_path,
        nn.BCEWithLogitsLoss(),
        eval_every=2,
        history_save_every=3,
        extras={"quantum_device": "lightning.qubit"},
    )

    history = trainer.train()

    assert history.spec == trainer.spec
    assert history.run_id == run_id(trainer.spec)
    assert history.extras == {
        "eval_every": 2,
        "history_save_every": 3,
        "quantum_device": "lightning.qubit",
    }


def test_training_writes_the_config_file_next_to_the_histories(
    tmp_path: Path,
) -> None:
    """Validates that a run records its configuration, so a person (or the run
    listing) can see what a config directory holds."""
    trainer, history_path = _build_history_trainer(
        tmp_path, nn.BCEWithLogitsLoss(), eval_every=1
    )

    trainer.train()

    config = json.loads((history_path.parent / "config.json").read_text())
    assert config["config_id"] == config_id(trainer.spec)
    assert config["spec"]["lr_step_size"] == HISTORY_STEP_SIZE
    assert "seed" not in config["spec"]


def test_last_epoch_is_always_evaluated_without_an_extra_final_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates that an epoch off the `eval_every` boundary still gets a
    validation value when it is the last one, and that the final report reuses
    it, so the validation loader is evaluated exactly once per recorded point."""
    trainer, _ = _build_history_trainer(
        tmp_path, nn.BCEWithLogitsLoss(), eval_every=2, epochs=5
    )
    evaluated_loaders: list[DataLoader] = []
    original_evaluate = trainer._evaluate_loader

    def spy(loader: DataLoader) -> tuple[float, float]:
        evaluated_loaders.append(loader)
        return original_evaluate(loader)

    monkeypatch.setattr(trainer, "_evaluate_loader", spy)

    history = trainer.train()

    assert [r.val_loss is not None for r in history.epochs] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert len(evaluated_loaders) == 3


@pytest.mark.parametrize(
    ("epochs", "expected_saved_lengths"),
    [(5, [2, 4, 5]), (4, [2, 4])],
)
def test_history_saved_every_n_epochs_and_once_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    epochs: int,
    expected_saved_lengths: list[int],
) -> None:
    """Validates that the history is written at each `history_save_every`
    boundary and once more on exit only when epochs remain unsaved, so a run
    ending on a boundary does not write twice."""
    trainer, _ = _build_history_trainer(
        tmp_path,
        nn.BCEWithLogitsLoss(),
        eval_every=100,
        history_save_every=2,
        epochs=epochs,
    )
    saved_lengths: list[int] = []
    original_save = trainer._save_history

    def spy(history: TrainingHistory) -> None:
        saved_lengths.append(len(history.epochs))
        original_save(history)

    monkeypatch.setattr(trainer, "_save_history", spy)

    trainer.train()

    assert saved_lengths == expected_saved_lengths


@pytest.mark.parametrize(
    ("eval_every", "history_save_every"),
    [(0, 10), (-1, 10), (1, 0), (1, -5)],
)
def test_non_positive_cadence_is_rejected(
    tmp_path: Path,
    eval_every: int,
    history_save_every: int,
) -> None:
    """Validates that a zero or negative cadence fails at construction instead
    of raising a modulo-by-zero error mid-training."""
    with pytest.raises(ValueError, match="must be >= 1"):
        _build_history_trainer(
            tmp_path,
            nn.BCEWithLogitsLoss(),
            eval_every=eval_every,
            history_save_every=history_save_every,
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"lr": 5e-4}, "lr"),
        ({"batch_size": 8}, "batch_size"),
        ({"lr_step_size": 3}, "lr_step_size"),
        ({"lr_gamma": 0.9}, "lr_gamma"),
    ],
)
def test_spec_that_disagrees_with_what_runs_is_rejected(
    tmp_path: Path, overrides: dict, field: str
) -> None:
    """Validates that a spec describing another learning rate, batch size, or
    schedule than the one that will run is refused, because the hash IDs would
    otherwise name a configuration that never ran."""
    spec = _spec(**{"lr_step_size": HISTORY_STEP_SIZE, "epochs": 2, **overrides})

    with pytest.raises(ValueError, match=field):
        _build_history_trainer(
            tmp_path, nn.BCEWithLogitsLoss(), eval_every=1, spec=spec
        )


def test_trainer_has_no_test_loader_argument(tmp_path: Path) -> None:
    """Validates that the trainer cannot be given the test cohort: it accepts a
    validation loader only, so training and per-epoch evaluation never see test
    data."""
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.tensor([0, 1] * 4)), batch_size=4
    )
    model = nn.Linear(4, 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with pytest.raises(TypeError, match="test_loader"):
        BaselineTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,  # type: ignore[call-arg]
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA),
            device=torch.device("cpu"),
            logger=setup_logger("test_trainer_no_test"),
            spec=_spec(),
            layout=ResultsLayout(str(tmp_path)),
        )


def test_history_survives_mid_run_crash(tmp_path: Path) -> None:
    """Validates that a crash in epoch 3 leaves a parseable JSON holding the two
    completed epochs and no leftover temporary file."""
    criterion = _FailingCriterion(max_calls=2 * BATCHES_PER_EPOCH)
    trainer, history_path = _build_history_trainer(tmp_path, criterion, eval_every=100)

    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer.train()

    saved = MetricsAnalyzer.load_history(str(history_path))
    assert [r.epoch for r in saved.epochs] == [1, 2]
    assert not history_path.with_name(f"{history_path.name}.tmp").exists()
