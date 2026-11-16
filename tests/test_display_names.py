"""Unit tests for the presentation names of the comparison plots.

Regression coverage
-------------------
- Configurations that differ in the variational block or in re-uploading got
  the same short name, so a presentation legend could not tell them apart.
"""

from dementia_boost.viz.display_names import production_label, production_labels


def _spec(**fields) -> dict:
    """A serialised Qiskit QTL spec with the fields the names read."""
    return {
        "paradigm": "qiskit_qtl",
        "ansatz": "paper",
        "block": None,
        "reupload": None,
        **fields,
    }


class TestProductionLabel:
    """Validates the short name of one configuration."""

    def test_default_block_names_only_paradigm_and_embedding(self) -> None:
        """The base paper's block without re-uploading adds nothing."""
        assert production_label(_spec()) == "QTL (Qiskit), Angle Embedding"

    def test_block_and_reuploading_are_named_when_set(self) -> None:
        """A non-default block and re-uploading appear in the name."""
        assert production_label(_spec(block="other", reupload=True)) == (
            "QTL (Qiskit), Angle Embedding, other, Re-uploading"
        )

    def test_classical_heads_have_no_quantum_part(self) -> None:
        """Baseline and CTL specs carry no embedding."""
        assert production_label({"paradigm": "ctl", "ansatz": None}) == "CTL"


def test_configurations_differing_in_reuploading_get_distinct_names() -> None:
    """Two configurations that differ only in re-uploading keep their names
    apart without falling back to the configuration ID."""
    names = production_labels({"a": _spec(), "b": _spec(reupload=True)})
    assert names["a"] != names["b"]
    assert "[" not in names["a"] + names["b"]
