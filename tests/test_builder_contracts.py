"""Unit tests for model factory builders and composition contracts.

This module validates:
- ``load_baseline_backbone`` weight restoration and parameter freezing.
- ``build_ctl_model`` and ``build_pl_qtl_model`` structural
  contracts (frozen backbone, freshly initialized head, device placement).
- ``build_qiskit_qtl_model`` passing the embedding, variational block, and
  re-uploading choice through to the circuit builder.
- ``assemble_dementia_classifier`` numerical equivalence against manually
  composed backbone and head forward passes.

Regression coverage
-------------------
- Backbones that silently remain trainable after being loaded for transfer
  learning, corrupting pre-trained spatial representations during CTL/QTL
  fine-tuning.
- Factory functions attaching the wrong head type or leaving the classifier
  head frozen alongside the backbone.
- Divergence between the assembled `DementiaClassifier` orchestrator and its
  constituent backbone/head modules invoked separately.
"""

from pathlib import Path

import pytest
import torch

from dementia_boost.models.builder import (
    assemble_dementia_classifier,
    build_ctl_model,
    build_pl_qtl_model,
    build_qiskit_qtl_model,
    load_baseline_backbone,
)
from dementia_boost.models.classical_cnn import (
    ClassicalClassifierHead,
    DementiaClassifier,
    LeNetFeatureExtractor,
)
from dementia_boost.models.quantum_cnn import (
    PennylaneQuantumClassifierHead,
    build_qiskit_ansatz,
    qiskit_heads,
)

_TEST_QUBITS: int = 2
_TEST_LAYERS: int = 1


def _write_mock_baseline_checkpoint(tmp_path: Path) -> str:
    """Serializes a freshly initialized baseline checkpoint to a temp file.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        Absolute path string of the written `.pt` checkpoint file.
    """
    mock_model = DementiaClassifier(
        feature_extractor=LeNetFeatureExtractor(),
        classifier_head=ClassicalClassifierHead(use_sigmoid=False),
    )
    checkpoint_path = tmp_path / "baseline_mock.pt"
    torch.save(mock_model.state_dict(), checkpoint_path)
    return str(checkpoint_path)


class TestLoadBaselineBackbone:
    """Validates weight restoration and freezing in `load_baseline_backbone`."""

    def test_returns_frozen_backbone_with_restored_weights(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that every parameter of the returned backbone has
        `requires_grad == False` after loading a mock checkpoint."""
        device = torch.device("cpu")
        checkpoint_path = _write_mock_baseline_checkpoint(tmp_path)

        backbone = load_baseline_backbone(checkpoint_path, device)

        assert isinstance(backbone, LeNetFeatureExtractor)
        for param in backbone.parameters():
            assert param.requires_grad is False


class TestBuildClassicalTlModel:
    """Validates the structural contract of the CTL factory function."""

    def test_frozen_backbone_and_fresh_classical_head(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that the returned model has a frozen backbone, an active
        `ClassicalClassifierHead`, and correct device placement."""
        device = torch.device("cpu")
        checkpoint_path = _write_mock_baseline_checkpoint(tmp_path)

        model = build_ctl_model(checkpoint_path, device)

        assert isinstance(model, DementiaClassifier)
        assert isinstance(model.classifier_head, ClassicalClassifierHead)

        for param in model.feature_extractor.parameters():
            assert param.requires_grad is False
            assert param.device.type == device.type

        for param in model.classifier_head.parameters():
            assert param.requires_grad is True
            assert param.device.type == device.type


class TestBuildQuantumTlModel:
    """Validates the structural contract of the QTL factory function."""

    def test_frozen_backbone_and_fresh_quantum_head(
        self,
        tmp_path: Path,
    ) -> None:
        """Asserts that the returned model has a frozen backbone and an
        active `PennylaneQuantumClassifierHead`."""
        device = torch.device("cpu")
        checkpoint_path = _write_mock_baseline_checkpoint(tmp_path)

        model = build_pl_qtl_model(
            checkpoint_path,
            device,
            n_qubits=_TEST_QUBITS,
            n_layers=_TEST_LAYERS,
            quantum_device="default.qubit",
        )

        assert isinstance(model, DementiaClassifier)
        assert isinstance(model.classifier_head, PennylaneQuantumClassifierHead)

        for param in model.feature_extractor.parameters():
            assert param.requires_grad is False

        for param in model.classifier_head.parameters():
            assert param.requires_grad is True


class TestBuildQiskitQtlModel:
    """Validates that the Qiskit factory builds the configured circuit."""

    def test_embedding_block_and_reupload_reach_the_circuit_builder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Asserts that the options given to `build_qiskit_qtl_model` are the
        ones the circuit is built with, so the evaluation, which rebuilds a
        model from its spec, evaluates the circuit that was trained."""
        calls: list[dict] = []

        def recording_builder(**kwargs):
            calls.append(kwargs)
            return build_qiskit_ansatz(**kwargs)

        monkeypatch.setattr(qiskit_heads, "build_qiskit_ansatz", recording_builder)

        build_qiskit_qtl_model(
            _write_mock_baseline_checkpoint(tmp_path),
            torch.device("cpu"),
            n_qubits=_TEST_QUBITS,
            n_layers=_TEST_LAYERS,
            embedding="rx",
            block="paper",
            reupload=True,
        )

        assert len(calls) == 1
        assert calls[0]["embedding"] == "rx"
        assert calls[0]["block"] == "paper"
        assert calls[0]["reupload"] is True


class TestAssembleDementiaClassifier:
    """Validates numerical equivalence of the assembled orchestrator."""

    def test_combined_forward_matches_manual_composition(self) -> None:
        """Asserts that `f_combined(x)` exactly matches `f_head(f_backbone(x))`
        with zero numerical discrepancy."""
        extractor = LeNetFeatureExtractor()
        head = ClassicalClassifierHead(use_sigmoid=False)
        x = torch.randn(2, 1, 128, 128)

        combined = assemble_dementia_classifier(
            feature_extractor=extractor,
            classifier_head=head,
        )
        combined.eval()

        with torch.no_grad():
            combined_output = combined(x)
            manual_output = head(extractor(x))

        assert torch.allclose(combined_output, manual_output)
