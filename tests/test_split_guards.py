"""Tests for the split integrity guard and the cohort-aware data loader.

Regression coverage
-------------------
- Nothing checked, after the split or before loading, that the cohorts on disk
  were disjoint and matched the split that was meant to be written, so a stale
  directory could be trained on silently.
- `OasisDataset` listed files in filesystem order.
"""

import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from dementia_boost.data.data_loader import OasisDataLoader
from dementia_boost.data.dataset import OasisDataset
from dementia_boost.data.split_guards import SplitIntegrityError, verify_split_layout


class TestVerifySplitLayout:
    """Validates the checks of `verify_split_layout` against the manifest."""

    def test_clean_layout_passes(self, tmp_path: Path, make_layout) -> None:
        """A layout that matches its manifest is accepted."""
        make_layout(tmp_path)
        verify_split_layout(str(tmp_path))

    def test_missing_manifest_raises_file_not_found(self, tmp_path: Path) -> None:
        """Without a manifest the layout cannot be verified, so it is refused."""
        with pytest.raises(FileNotFoundError, match="split_manifest"):
            verify_split_layout(str(tmp_path))

    def test_subject_in_two_cohorts_raises_with_its_id(
        self, tmp_path: Path, make_layout
    ) -> None:
        """A train subject also present in test is reported by ID."""
        make_layout(tmp_path)
        stale = tmp_path / "train" / "OAS2_0001_MR1_mpr-1.pt"
        shutil.copy(stale, tmp_path / "test" / stale.name)

        with pytest.raises(SplitIntegrityError, match="OAS2_0001"):
            verify_split_layout(str(tmp_path))

    def test_file_in_the_wrong_cohort_raises(self, tmp_path: Path, make_layout) -> None:
        """A file that sits in a different cohort than the manifest says is refused."""
        make_layout(tmp_path)
        moved = tmp_path / "train" / "OAS2_0002_MR1_mpr-1.pt"
        shutil.move(moved, tmp_path / "val" / moved.name)

        with pytest.raises(SplitIntegrityError, match="OAS2_0002"):
            verify_split_layout(str(tmp_path))

    def test_subject_unknown_to_the_manifest_raises(
        self, tmp_path: Path, make_layout
    ) -> None:
        """A file of a subject the manifest never assigned is refused."""
        make_layout(tmp_path)
        torch.save((torch.randn(1, 8, 8), 0), tmp_path / "test" / "OAS2_0099_MR1_x.pt")

        with pytest.raises(SplitIntegrityError, match="OAS2_0099"):
            verify_split_layout(str(tmp_path))

    def test_missing_file_raises(self, tmp_path: Path, make_layout) -> None:
        """A subject with fewer files than the manifest records is refused."""
        make_layout(tmp_path)
        (tmp_path / "val" / "OAS2_0005_MR2_mpr-1.pt").unlink()

        with pytest.raises(SplitIntegrityError, match="OAS2_0005"):
            verify_split_layout(str(tmp_path))

    def test_cohort_with_one_class_raises(self, tmp_path: Path, make_layout) -> None:
        """A manifest whose cohort holds a single class is refused."""
        make_layout(tmp_path, {"train": [1, 2, 3, 4], "val": [5, 6], "test": [1 + 6]})

        with pytest.raises(SplitIntegrityError, match="test"):
            verify_split_layout(str(tmp_path))


class TestOasisDataLoaderCohorts:
    """Validates cohort selection and the integrity check of the loader."""

    @pytest.fixture
    def loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> OasisDataLoader:
        """A loader pointing at an empty temporary results directory."""
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        return OasisDataLoader(batch_size=4)

    def test_loads_every_cohort_from_a_clean_layout(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """Train, val, and test loaders are returned for a clean layout."""
        make_layout(tmp_path)
        sizes = {
            name: len(loader.get_data_loader(name).dataset)  # type: ignore[arg-type]
            for name in ("train", "val", "test")
        }
        assert sizes == {"train": 8, "val": 4, "test": 4}

    def test_only_train_is_shuffled(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """Training shuffles; validation and test keep a fixed order."""
        make_layout(tmp_path)
        assert isinstance(loader.get_data_loader("train").sampler, RandomSampler)
        assert isinstance(loader.get_data_loader("val").sampler, SequentialSampler)
        assert isinstance(loader.get_data_loader("test").sampler, SequentialSampler)

    def test_unknown_cohort_raises(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """A cohort name outside train, val, and test is rejected."""
        make_layout(tmp_path)
        with pytest.raises(ValueError, match="cohort"):
            loader.get_data_loader("holdout")  # type: ignore[arg-type]

    def test_leaky_layout_is_refused(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """A subject shared by two cohorts stops the loader, naming the subject."""
        make_layout(tmp_path)
        stale = tmp_path / "train" / "OAS2_0003_MR1_mpr-1.pt"
        shutil.copy(stale, tmp_path / "test" / stale.name)

        with pytest.raises(SplitIntegrityError, match="OAS2_0003"):
            loader.get_data_loader("train")

    def test_layout_that_disagrees_with_manifest_is_refused(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """A missing exam file stops the loader even if no subject is shared."""
        make_layout(tmp_path)
        (tmp_path / "test" / "OAS2_0007_MR1_mpr-1.pt").unlink()

        with pytest.raises(SplitIntegrityError, match="OAS2_0007"):
            loader.get_data_loader("test")

    def test_missing_manifest_is_refused(
        self, tmp_path: Path, make_layout, loader: OasisDataLoader
    ) -> None:
        """Cohort files without a manifest cannot be verified, so they are refused."""
        make_layout(tmp_path)
        (tmp_path / "split_manifest.json").unlink()

        with pytest.raises(FileNotFoundError):
            loader.get_data_loader("train")


def test_dataset_lists_files_in_sorted_order(tmp_path: Path) -> None:
    """Sample order must not depend on the filesystem listing order."""
    names = [f"s_{i:02d}.pt" for i in (7, 2, 11, 0, 9, 4, 1, 10, 5, 3, 8, 6)]
    for name in names:
        torch.save((torch.zeros(1, 4, 4), 0), tmp_path / name)

    dataset = OasisDataset(directory_path=str(tmp_path))

    assert [Path(p).name for p in dataset.file_list] == sorted(names)
