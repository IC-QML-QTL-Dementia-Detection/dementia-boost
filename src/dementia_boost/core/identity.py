"""Run identity: the typed run specification, its hash IDs, and its label.

A run is described by a `RunSpec`. Two IDs are derived from it by hashing:

- `config_id` hashes every field except the seed, so the seeds of one
  configuration share it.
- `run_id` hashes every field, seed included.

IDs are opaque: nothing recovers configuration from a name. The spec itself is
stored next to the results, and the label is a readable rendering of it for
plots and logs. Fields that are `None` are left out of the hash, so adding a new
optional field later does not change any existing ID.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

ID_LENGTH = 12


class Paradigm(StrEnum):
    """The training paradigms, as the single definition of their names."""

    BASELINE = "baseline"
    CTL = "ctl"
    QTL = "qtl"
    QISKIT_QTL = "qiskit_qtl"


_HEAD_FIELDS = (
    "ansatz",
    "n_qubits",
    "n_layers",
    "quantum_lr",
    "gradient",
    "spsa_epsilon",
    "backbone_id",
)
_REQUIRED: dict[Paradigm, tuple[str, ...]] = {
    Paradigm.BASELINE: (),
    Paradigm.CTL: ("backbone_id",),
    Paradigm.QTL: ("ansatz", "n_qubits", "n_layers", "gradient", "backbone_id"),
    Paradigm.QISKIT_QTL: (
        "ansatz",
        "n_qubits",
        "n_layers",
        "gradient",
        "spsa_epsilon",
        "backbone_id",
    ),
}
_FORBIDDEN: dict[Paradigm, tuple[str, ...]] = {
    Paradigm.BASELINE: _HEAD_FIELDS,
    Paradigm.CTL: (
        "ansatz",
        "n_qubits",
        "n_layers",
        "quantum_lr",
        "gradient",
        "spsa_epsilon",
    ),
    Paradigm.QTL: ("spsa_epsilon",),
    Paradigm.QISKIT_QTL: (),
}
_FLOAT_FIELDS = ("lr", "lr_gamma", "quantum_lr", "spsa_epsilon")


def short_hash(payload: Mapping[str, Any]) -> str:
    """Hashes a mapping into a short, stable identifier.

    The mapping is serialised as JSON with sorted keys and compact separators
    after dropping `None` values, then hashed with SHA-256.

    Args:
        payload: JSON-serialisable mapping.

    Returns:
        The first `ID_LENGTH` lowercase hex characters of the digest.
    """
    kept = {key: value for key, value in payload.items() if value is not None}
    text = json.dumps(kept, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:ID_LENGTH]


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines a training run.

    Fields that do not apply to a paradigm must be `None`, and fields it needs
    must be set; this is checked on construction.

    Attributes:
        paradigm: The training paradigm.
        lr: Learning rate of the classical parameters.
        lr_step_size: Epochs between `StepLR` decays.
        lr_gamma: `StepLR` decay factor.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        split_id: ID of the data split the run was trained on.
        seed: Random seed of the run.
        ansatz: Name of the variational circuit (quantum heads only).
        n_qubits: Number of qubits (quantum heads only).
        n_layers: Number of ansatz layers (quantum heads only).
        quantum_lr: Learning rate of the circuit weights, or `None` to share `lr`
            (quantum heads only).
        gradient: Gradient method, for example "adjoint" or "spsa" (quantum heads
            only).
        spsa_epsilon: SPSA perturbation size (Qiskit head only).
        backbone_id: `run_id` of the baseline the head was built on (heads only).
    """

    paradigm: Paradigm
    lr: float
    lr_step_size: int
    lr_gamma: float
    epochs: int
    batch_size: int
    split_id: str
    seed: int
    ansatz: str | None = None
    n_qubits: int | None = None
    n_layers: int | None = None
    quantum_lr: float | None = None
    gradient: str | None = None
    spsa_epsilon: float | None = None
    backbone_id: str | None = None

    def __post_init__(self) -> None:
        """Normalises the paradigm and float fields, then validates the spec.

        Raises:
            ValueError: If a field the paradigm needs is `None`, or a field it
                does not use is set.
        """
        object.__setattr__(self, "paradigm", Paradigm(self.paradigm))
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

        paradigm = self.paradigm.value
        for name in _REQUIRED[self.paradigm]:
            if getattr(self, name) is None:
                raise ValueError(f"A {paradigm} spec requires {name}.")
        for name in _FORBIDDEN[self.paradigm]:
            if getattr(self, name) is not None:
                raise ValueError(f"A {paradigm} spec must not set {name}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialises the spec, every field included (`None` ones too).

        Returns:
            A JSON-serialisable dictionary, with the paradigm as its name.
        """
        data = asdict(self)
        data["paradigm"] = self.paradigm.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSpec":
        """Rebuilds a spec from `to_dict` output, in any key order.

        Args:
            data: Mapping with the spec's fields.

        Returns:
            The validated spec.
        """
        return cls(**data)


def config_id(spec: RunSpec) -> str:
    """Hashes every field of the spec except the seed.

    Args:
        spec: The run specification.

    Returns:
        The ID shared by all seeds of one configuration.
    """
    payload = spec.to_dict()
    del payload["seed"]
    return short_hash(payload)


def run_id(spec: RunSpec) -> str:
    """Hashes every field of the spec, the seed included.

    Args:
        spec: The run specification.

    Returns:
        The ID of this one run.
    """
    return short_hash(spec.to_dict())


def backbone_id_of(baseline: RunSpec) -> str:
    """Returns the ID that heads built on a baseline record as their backbone.

    Args:
        baseline: The spec of a baseline run.

    Returns:
        The baseline's `run_id`.

    Raises:
        ValueError: If the spec is not a baseline.
    """
    if baseline.paradigm is not Paradigm.BASELINE:
        raise ValueError(
            f"Only a baseline spec can be a backbone, got {baseline.paradigm.value}."
        )
    return run_id(baseline)


def label(spec: RunSpec, include_seed: bool = True) -> str:
    """Renders the spec as a short readable string for plots and logs.

    The label is output only; nothing parses it.

    Args:
        spec: The run specification.
        include_seed: Whether to end the label with the seed. Leave it out for
            legends that cover all seeds of a configuration.

    Returns:
        For example `qtl | paper | 6q x 4L | lr 0.0001 | seed 3`.
    """
    parts = [spec.paradigm.value]
    if spec.ansatz is not None:
        parts.append(spec.ansatz)
    if spec.n_qubits is not None:
        parts.append(f"{spec.n_qubits}q x {spec.n_layers}L")
    parts.append(f"lr {spec.lr:g}")
    if spec.quantum_lr is not None:
        parts.append(f"qlr {spec.quantum_lr:g}")
    if include_seed:
        parts.append(f"seed {spec.seed}")
    return " | ".join(parts)
