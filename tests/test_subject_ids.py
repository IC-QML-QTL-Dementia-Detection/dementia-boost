"""Unit tests for canonical subject IDs, label mapping, and manual overrides.

Regression coverage
-------------------
- Manual overrides were neither validated nor checked for conflicts, so an
  unknown ID was silently ignored and one ID could sit in both cohorts.
- Spelling and case variants of one subject were treated as different subjects,
  and conflicting labels were resolved silently.
"""

import pandas as pd
import pytest

from dementia_boost.data.subject_ids import (
    build_subject_labels,
    canonical_label,
    canonical_subject_id,
    subject_of_visit,
    validate_overrides,
)


class TestCanonicalSubjectId:
    """Validates the single canonical form `OAS2_NNNN`."""

    @pytest.mark.parametrize("raw", ["OAS2_0001", " oas2_0001 ", "Oas2_0001"])
    def test_variants_map_to_one_id(self, raw: str) -> None:
        """Case and whitespace must not change the ID."""
        assert canonical_subject_id(raw) == "OAS2_0001"

    @pytest.mark.parametrize(
        "raw", ["", "OAS2_1", "OAS3_0001", "0001", "OAS2_00012", "OAS2_0001_MR2"]
    )
    def test_malformed_id_raises(self, raw: str) -> None:
        """Anything but `OAS2_` plus four digits is rejected, visit IDs included.

        A visit ID names one exam, not a patient, so it is never accepted where
        a subject is expected (for example in a manual override).
        """
        with pytest.raises(ValueError):
            canonical_subject_id(raw)


class TestSubjectOfVisit:
    """Validates the mapping from an exam (MRI ID) to its patient."""

    @pytest.mark.parametrize("raw", ["OAS2_0001_MR2", " oas2_0001_mr1 "])
    def test_visit_maps_to_its_subject(self, raw: str) -> None:
        """Every exam of a patient maps to that patient's subject ID."""
        assert subject_of_visit(raw) == "OAS2_0001"

    def test_subject_without_visit_suffix_raises(self) -> None:
        """A bare subject ID is not an exam ID."""
        with pytest.raises(ValueError):
            subject_of_visit("OAS2_0001")


class TestCanonicalLabel:
    """Validates case-insensitive label mapping."""

    @pytest.mark.parametrize("raw", ["Nondemented", "NonDemented", " nondemented "])
    def test_nondemented_spellings_map_to_zero(self, raw: str) -> None:
        """Every spelling of the negative class must map to 0."""
        assert canonical_label(raw) == 0

    @pytest.mark.parametrize("raw", ["Demented", "demented", " DEMENTED"])
    def test_demented_spellings_map_to_one(self, raw: str) -> None:
        """Every spelling of the positive class must map to 1."""
        assert canonical_label(raw) == 1

    def test_converted_is_not_a_label(self) -> None:
        """`Converted` is an exclusion, so it maps to None instead of a class."""
        assert canonical_label("Converted") is None

    def test_unknown_label_raises(self) -> None:
        """An unrecognised group name must raise instead of being skipped."""
        with pytest.raises(ValueError):
            canonical_label("Unknown")


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Builds a metadata frame from (Subject ID, Group) pairs."""
    return pd.DataFrame(rows, columns=["Subject ID", "Group"])


class TestBuildSubjectLabels:
    """Validates subject-level label building and recorded exclusions."""

    def test_visits_of_one_subject_collapse_to_one_entry(self) -> None:
        """Several rows of one subject, in any spelling, give one label."""
        labels, _ = build_subject_labels(
            _frame([("OAS2_0001", "Demented"), (" oas2_0001", "demented")])
        )
        assert labels == {"OAS2_0001": 1}

    def test_converted_subject_is_excluded_with_reason(self) -> None:
        """A `Converted` subject leaves the labels and is listed with a reason."""
        labels, excluded = build_subject_labels(
            _frame([("OAS2_0001", "Demented"), ("OAS2_0003", "Converted")])
        )
        assert "OAS2_0003" not in labels
        assert "converted" in excluded["OAS2_0003"].lower()

    def test_conflicting_labels_raise(self) -> None:
        """One subject with two different classes is a data error."""
        with pytest.raises(ValueError, match="OAS2_0001"):
            build_subject_labels(
                _frame([("OAS2_0001", "Demented"), ("OAS2_0001", "Nondemented")])
            )


class TestValidateOverrides:
    """Validates canonicalisation and rejection rules for manual overrides."""

    _KNOWN = {"OAS2_0001", "OAS2_0002", "OAS2_0003"}

    def test_overrides_are_canonicalised_and_sorted(self) -> None:
        """Overrides come back in canonical form, sorted."""
        train, test = validate_overrides(
            [" oas2_0002", "OAS2_0001"], ["oas2_0003"], self._KNOWN
        )
        assert train == ["OAS2_0001", "OAS2_0002"]
        assert test == ["OAS2_0003"]

    def test_visit_id_override_raises(self) -> None:
        """An exam ID must not silently move a whole patient."""
        with pytest.raises(ValueError):
            validate_overrides(["OAS2_0001_MR2"], [], self._KNOWN)

    def test_id_in_both_lists_raises(self) -> None:
        """An ID forced into both cohorts must raise and name the ID."""
        with pytest.raises(ValueError, match="OAS2_0001"):
            validate_overrides(["OAS2_0001"], ["oas2_0001"], self._KNOWN)

    def test_unknown_id_raises(self) -> None:
        """An ID that is not a known labelled subject must raise."""
        with pytest.raises(ValueError, match="OAS2_0099"):
            validate_overrides(["OAS2_0099"], [], self._KNOWN)
