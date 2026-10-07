"""Unit tests for run identity: the spec, its hash IDs, and the label.

Regression coverage
-------------------
- Identity was carried by names (`qtl_seed_3`), so two runs that differed in
  qubits, layers, or learning rate but not in seed collided on disk and in the
  "skip if exists" check.
- Configuration was recovered by parsing file names.
"""

import dataclasses
import json
import re

import pytest

from dementia_boost.core.identity import (
    ID_LENGTH,
    Paradigm,
    RunSpec,
    backbone_id_of,
    config_id,
    label,
    run_id,
    short_hash,
)
from dementia_boost.data.split import SubjectSplit
from dementia_boost.data.split_manifest import compute_split_id

_SPLIT_ID = "5192b1b7c0d3"
_BACKBONE_ID = "c8375944e970"


def _baseline(**overrides) -> RunSpec:
    """A valid baseline spec, with fields replaceable through keyword arguments."""
    fields = {
        "paradigm": Paradigm.BASELINE,
        "lr": 1e-4,
        "lr_step_size": 10,
        "lr_gamma": 0.75,
        "epochs": 100,
        "batch_size": 64,
        "split_id": _SPLIT_ID,
        "seed": 17,
    }
    return RunSpec(**{**fields, **overrides})


def _ctl(**overrides) -> RunSpec:
    """A valid classical transfer learning spec."""
    return _baseline(
        **{
            "paradigm": Paradigm.CTL,
            "backbone_id": _BACKBONE_ID,
            "seed": 5,
            **overrides,
        }
    )


def _qtl(**overrides) -> RunSpec:
    """A valid PennyLane quantum transfer learning spec (6 qubits, 4 layers)."""
    return _baseline(
        **{
            "paradigm": Paradigm.QTL,
            "ansatz": "paper",
            "n_qubits": 6,
            "n_layers": 4,
            "gradient": "adjoint",
            "backbone_id": _BACKBONE_ID,
            "seed": 3,
            **overrides,
        }
    )


def _qiskit(**overrides) -> RunSpec:
    """A valid Qiskit quantum transfer learning spec."""
    return _qtl(
        **{
            "paradigm": Paradigm.QISKIT_QTL,
            "gradient": "spsa",
            "spsa_epsilon": 0.1,
            **overrides,
        }
    )


class TestParadigm:
    """Validates the single definition of paradigm names."""

    def test_members_and_values(self) -> None:
        """The four paradigms exist under their current names."""
        assert {p.value for p in Paradigm} == {"baseline", "ctl", "qtl", "qiskit_qtl"}

    def test_is_a_string(self) -> None:
        """A paradigm compares equal to its name, so it can sit in a path or JSON."""
        assert Paradigm.QTL == "qtl"
        assert Paradigm("ctl") is Paradigm.CTL


class TestShortHash:
    """Validates the shared hash helper."""

    def test_length_and_alphabet(self) -> None:
        """The hash is `ID_LENGTH` lowercase hex characters."""
        assert re.fullmatch(f"[0-9a-f]{{{ID_LENGTH}}}", short_hash({"a": 1}))

    def test_key_order_does_not_matter(self) -> None:
        """The same mapping in another order gives the same hash."""
        assert short_hash({"a": 1, "b": 2}) == short_hash({"b": 2, "a": 1})

    def test_none_values_are_dropped(self) -> None:
        """A `None` field does not change the hash, so adding an optional field
        later keeps every existing ID."""
        assert short_hash({"a": 1, "b": None}) == short_hash({"a": 1})

    def test_split_id_is_unchanged_by_the_shared_helper(self) -> None:
        """`split_id` keeps the value it had before it used `short_hash`."""
        split = SubjectSplit(train=["a", "b"], val=["c"], test=["d"])
        assert compute_split_id(split) == "c5288af423aa"


class TestIdentity:
    """Validates `config_id` and `run_id`."""

    def test_golden_values(self) -> None:
        """Fixed spec, fixed IDs: guards the canonical serialisation. The values
        were computed independently of the implementation."""
        spec = _qtl()
        assert config_id(spec) == "3976415a4171"
        assert run_id(spec) == "62753438444d"

    def test_same_spec_same_ids(self) -> None:
        """Two equal specs give equal IDs."""
        assert run_id(_qtl()) == run_id(_qtl())
        assert config_id(_qtl()) == config_id(_qtl())

    _CHANGES = {
        "ansatz": "other",
        "n_qubits": 8,
        "n_layers": 3,
        "lr": 1e-3,
        "quantum_lr": 1e-2,
        "lr_step_size": 5,
        "lr_gamma": 0.5,
        "epochs": 50,
        "batch_size": 32,
        "gradient": "parameter_shift",
        "backbone_id": "a" * ID_LENGTH,
        "split_id": "b" * ID_LENGTH,
    }

    @pytest.mark.parametrize("field", sorted(_CHANGES))
    def test_any_field_change_changes_both_ids(self, field: str) -> None:
        """Changing one field (other than the seed) gives a new run and config."""
        changed = dataclasses.replace(_qtl(), **{field: self._CHANGES[field]})
        assert run_id(changed) != run_id(_qtl())
        assert config_id(changed) != config_id(_qtl())

    def test_seed_changes_run_id_but_not_config_id(self) -> None:
        """Seeds of one configuration share a `config_id` and differ in `run_id`."""
        other = _qtl(seed=4)
        assert config_id(other) == config_id(_qtl())
        assert run_id(other) != run_id(_qtl())

    def test_key_order_and_int_versus_float_do_not_matter(self) -> None:
        """A spec rebuilt from a reordered dict, with `1` where a float is
        expected, has the same IDs."""
        data = _baseline(lr=1.0).to_dict()
        reordered = dict(reversed(list(data.items())))
        reordered["lr"] = 1
        rebuilt = RunSpec.from_dict(reordered)
        assert run_id(rebuilt) == run_id(_baseline(lr=1.0))
        assert config_id(rebuilt) == config_id(_baseline(lr=1.0))

    def test_quantum_lr_none_differs_from_an_explicit_value(self) -> None:
        """`None` means "share lr", which is not the same configuration as an
        explicit quantum learning rate equal to `lr`."""
        assert config_id(_qtl()) != config_id(_qtl(quantum_lr=1e-4))


class TestBackbone:
    """Validates the link from heads to the baseline they were built on."""

    def test_backbone_id_is_the_baseline_run_id(self) -> None:
        """The backbone of a head is the `run_id` of its baseline spec."""
        baseline = _baseline()
        assert backbone_id_of(baseline) == run_id(baseline)

    def test_a_different_backbone_is_a_different_configuration(self) -> None:
        """Training on another baseline gives another `config_id`."""
        first = _ctl(backbone_id=backbone_id_of(_baseline(seed=1)))
        second = _ctl(backbone_id=backbone_id_of(_baseline(seed=2)))
        assert config_id(first) != config_id(second)

    def test_only_a_baseline_can_be_a_backbone(self) -> None:
        """A head spec is not a backbone."""
        with pytest.raises(ValueError, match="baseline"):
            backbone_id_of(_ctl())


class TestSerialisation:
    """Validates the JSON round trip of the spec."""

    @pytest.mark.parametrize("make", [_baseline, _ctl, _qtl, _qiskit])
    def test_round_trip_through_json(self, make) -> None:
        """A spec written to JSON and read back is equal to the original."""
        spec = make()
        rebuilt = RunSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert rebuilt == spec

    def test_to_dict_lists_every_field(self) -> None:
        """The serialised form names every field, `None` ones included, so a
        stored spec is readable on its own."""
        assert set(_baseline().to_dict()) == {
            f.name for f in dataclasses.fields(RunSpec)
        }


class TestValidation:
    """Validates that fields that do not apply to a paradigm are `None`."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n_qubits": 6},
            {"n_layers": 4},
            {"ansatz": "paper"},
            {"quantum_lr": 1e-2},
            {"gradient": "adjoint"},
            {"backbone_id": _BACKBONE_ID},
        ],
    )
    def test_baseline_rejects_head_fields(self, overrides: dict) -> None:
        """A baseline has no quantum fields and no backbone."""
        with pytest.raises(ValueError, match="baseline"):
            _baseline(**overrides)

    def test_ctl_requires_a_backbone(self) -> None:
        """A transfer learning head needs its backbone."""
        with pytest.raises(ValueError, match="backbone_id"):
            _ctl(backbone_id=None)

    @pytest.mark.parametrize("overrides", [{"n_qubits": 6}, {"ansatz": "paper"}])
    def test_ctl_rejects_quantum_fields(self, overrides: dict) -> None:
        """A classical head has no quantum fields."""
        with pytest.raises(ValueError, match="ctl"):
            _ctl(**overrides)

    @pytest.mark.parametrize(
        "field", ["ansatz", "n_qubits", "n_layers", "gradient", "backbone_id"]
    )
    def test_qtl_requires_its_fields(self, field: str) -> None:
        """A PennyLane quantum head needs ansatz, qubits, layers, gradient, backbone."""
        with pytest.raises(ValueError, match=field):
            _qtl(**{field: None})

    def test_qtl_rejects_spsa_epsilon(self) -> None:
        """The SPSA step belongs to the Qiskit path only."""
        with pytest.raises(ValueError, match="spsa_epsilon"):
            _qtl(spsa_epsilon=0.1)

    def test_qiskit_requires_spsa_epsilon(self) -> None:
        """The Qiskit path records its SPSA step in the spec."""
        with pytest.raises(ValueError, match="spsa_epsilon"):
            _qiskit(spsa_epsilon=None)

    def test_a_string_paradigm_is_accepted(self) -> None:
        """A paradigm given as its name is converted to the enum."""
        assert _baseline(paradigm="baseline").paradigm is Paradigm.BASELINE


class TestLabel:
    """Validates the human-readable label."""

    def test_qtl_label(self) -> None:
        """The label names the paradigm, ansatz, size, learning rate, and seed."""
        assert label(_qtl()) == "qtl | paper | 6q x 4L | lr 0.0001 | seed 3"

    def test_label_without_seed_for_config_legends(self) -> None:
        """A config-level label leaves the seed out."""
        assert label(_qtl(), include_seed=False) == "qtl | paper | 6q x 4L | lr 0.0001"

    def test_quantum_lr_appears_when_set(self) -> None:
        """A separate quantum learning rate is shown."""
        assert "qlr 0.01" in label(_qtl(quantum_lr=1e-2))

    def test_classical_labels_have_no_quantum_part(self) -> None:
        """Baseline and CTL labels carry no qubits or ansatz."""
        assert label(_baseline()) == "baseline | lr 0.0001 | seed 17"
        assert label(_ctl()) == "ctl | lr 0.0001 | seed 5"

    def test_label_is_stable(self) -> None:
        """The same spec always gives the same label."""
        assert label(_qtl()) == label(_qtl())
