"""Unit tests for the NIfTI data engineering and patient isolation pipeline.

This module validates:
- ``OasisDataProcessor`` CSV parsing, subject splitting, and 3D-to-2D central
  axial slice extraction.
- ``MinMaxNormalize`` dynamic range scaling including the zero-division guard.
- ``OasisDataset`` tensor loading contracts.
- ``OasisDataLoader`` batch size, shuffle, and missing-path contracts.

Regression coverage
-------------------
- Corrupted ground-truth labels from incorrect CSV parsing or inclusion of
  'Converted' subjects.
- Patient-level data leakage across train/test splits.
- Silent NaN or Inf from dividing by zero on uniform-pixel scans.
- Broken DataLoader factories when required data paths are absent.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import RandomSampler, SequentialSampler

from dementia_boost.data.data_loader import MinMaxNormalize, OasisDataLoader
from dementia_boost.data.data_processor import OasisDataProcessor
from dementia_boost.data.dataset import OasisDataset

_CSV_HEADER = (
    "Subject ID,MRI ID,Group,Visit,MR Delay,"
    "M/F,Hand,Age,Educ,SES,MMSE,CDR,eTIV,nWBV,ASF\n"
)


def _make_csv(rows: list[str]) -> str:
    """Assembles a minimal OASIS-style CSV string from a list of row strings.

    Args:
        rows: List of CSV data row strings (without the header).

    Returns:
        A single string containing the header followed by the provided rows.
    """
    return _CSV_HEADER + "\n".join(rows)


class TestOasisDataProcessorParseCsv:
    """Validates label mapping and 'Converted' filtering in
    OasisDataProcessor._parse_csv."""

    def _write_csv(self, content: str, tmp_path: Path) -> str:
        """Writes CSV content to a temporary file and returns its absolute path.

        Args:
            content: Full CSV string to write.
            tmp_path: pytest-provided temporary directory.

        Returns:
            Absolute path string of the written CSV file.
        """
        csv_file = tmp_path / "oasis.csv"
        csv_file.write_text(content)
        return str(csv_file)

    def test_nondemented_maps_to_zero(self, tmp_path: Path) -> None:
        """Asserts that 'Nondemented' rows produce integer label 0."""
        content = _make_csv(
            [
                "OAS2_0001,OAS2_0001_MR1,Nondemented,1,0,M,R,87,2,2,30,0,1987,0.696,0.883",
            ]
        )
        processor = OasisDataProcessor(csv_path=self._write_csv(content, tmp_path))
        result = processor._parse_csv()
        assert result["OAS2_0001"] == 0

    def test_demented_maps_to_one(self, tmp_path: Path) -> None:
        """Asserts that 'Demented' rows produce integer label 1."""
        content = _make_csv(
            [
                "OAS2_0002,OAS2_0002_MR1,Demented,1,0,M,R,75,2,2,23,0.5,1678,0.736,0.894",
            ]
        )
        processor = OasisDataProcessor(csv_path=self._write_csv(content, tmp_path))
        result = processor._parse_csv()
        assert result["OAS2_0002"] == 1

    def test_converted_rows_are_excluded(self, tmp_path: Path) -> None:
        """Asserts that 'Converted' subjects are absent from the returned mapping."""
        content = _make_csv(
            [
                "OAS2_0001,OAS2_0001_MR1,Nondemented,1,0,M,R,87,2,2,30,0,1987,0.696,0.883",
                "OAS2_0003,OAS2_0003_MR1,Converted,1,0,F,R,70,3,2,26,0.5,1234,0.756,0.900",
            ]
        )
        processor = OasisDataProcessor(csv_path=self._write_csv(content, tmp_path))
        result = processor._parse_csv()
        assert "OAS2_0003" not in result
        assert "OAS2_0001" in result

    def test_mixed_labels_parsed_correctly(self, tmp_path: Path) -> None:
        """Asserts that a mixed CSV yields exactly the valid non-Converted entries."""
        content = _make_csv(
            [
                "OAS2_0001,OAS2_0001_MR1,Nondemented,1,0,M,R,87,2,2,30,0,1987,0.696,0.883",
                "OAS2_0002,OAS2_0002_MR1,Demented,1,0,M,R,75,2,2,23,0.5,1678,0.736,0.894",
                "OAS2_0003,OAS2_0003_MR1,Converted,1,0,F,R,70,3,2,26,0.5,1234,0.756,0.900",
            ]
        )
        processor = OasisDataProcessor(csv_path=self._write_csv(content, tmp_path))
        result = processor._parse_csv()
        assert result == {"OAS2_0001": 0, "OAS2_0002": 1}


class TestOasisDataProcessorSubjectSplit:
    """Validates patient-level disjointness and manual override pinning
    in _split_subjects."""

    def _make_processor(self, tmp_path: Path) -> OasisDataProcessor:
        """Returns an OasisDataProcessor pointing to a minimal placeholder CSV.

        Args:
            tmp_path: pytest-provided temporary directory.

        Returns:
            An OasisDataProcessor instance.
        """
        dummy = tmp_path / "dummy.csv"
        dummy.write_text(_CSV_HEADER)
        return OasisDataProcessor(csv_path=str(dummy))

    def _build_metadata(self, n: int) -> dict[str, int]:
        """Generates a synthetic subject metadata dict with n unique subjects.

        Args:
            n: Number of subjects to generate.

        Returns:
            A dict mapping subject IDs to alternating 0/1 labels.
        """
        return {f"OAS2_{i:04d}": i % 2 for i in range(1, n + 1)}

    def test_train_and_test_are_disjoint(self, tmp_path: Path) -> None:
        """Critical leakage check: train_subjects ∩ test_subjects must be empty."""
        processor = self._make_processor(tmp_path)
        metadata = self._build_metadata(20)
        train, test = processor._split_subjects(metadata, 0.7, 42, [], [])
        assert not (train & test)

    def test_union_covers_all_subjects(self, tmp_path: Path) -> None:
        """Every subject in metadata must appear in exactly one of the two splits."""
        processor = self._make_processor(tmp_path)
        metadata = self._build_metadata(20)
        train, test = processor._split_subjects(metadata, 0.7, 42, [], [])
        assert train | test == set(metadata.keys())

    def test_manual_train_ids_pinned_to_train(self, tmp_path: Path) -> None:
        """Subjects in manual_train must appear in train and not in test."""
        processor = self._make_processor(tmp_path)
        metadata = self._build_metadata(20)
        forced = ["OAS2_0001", "OAS2_0002"]
        train, test = processor._split_subjects(metadata, 0.7, 42, forced, [])
        for subj in forced:
            assert subj in train
            assert subj not in test

    def test_manual_test_ids_pinned_to_test(self, tmp_path: Path) -> None:
        """Subjects in manual_test must appear in test and not in train."""
        processor = self._make_processor(tmp_path)
        metadata = self._build_metadata(20)
        forced = ["OAS2_0003", "OAS2_0004"]
        train, test = processor._split_subjects(metadata, 0.7, 42, [], forced)
        for subj in forced:
            assert subj in test
            assert subj not in train

    def test_split_is_deterministic_for_identical_seeds(self, tmp_path: Path) -> None:
        """Two calls with the same seed must produce identical partitions."""
        processor = self._make_processor(tmp_path)
        metadata = self._build_metadata(30)
        train_a, test_a = processor._split_subjects(metadata, 0.7, 42, [], [])
        train_b, test_b = processor._split_subjects(metadata, 0.7, 42, [], [])
        assert train_a == train_b
        assert test_a == test_b


class TestOasisDataProcessorSliceExtraction:
    """Validates the 3D-to-2D central axial slice extraction via a synthetic
    NIfTI image."""

    def test_central_axial_slice_extracted_and_serialized(self, tmp_path: Path) -> None:
        """Asserts extraction of D // 2 slice with shape (1, H, W), saved as .pt.

        Constructs a synthetic NIfTI volume with known values, replicates the
        extraction logic from OasisDataProcessor._process_subset, and verifies
        the saved tensor against the expected central depth index.
        """
        D, H, W = 64, 96, 96
        volume_data = np.random.rand(D, H, W).astype(np.float32)
        nifti_img = nib.Nifti1Image(volume_data, np.eye(4))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        save_path = output_dir / "test_slice.pt"

        volume_squeezed = np.squeeze(nifti_img.get_fdata())
        assert volume_squeezed.ndim == 3

        middle_idx = volume_squeezed.shape[0] // 2
        slice_2d = volume_squeezed[middle_idx, :, :]
        tensor_volume = torch.from_numpy(slice_2d).float().unsqueeze(0)

        torch.save((tensor_volume, 1), str(save_path))

        assert save_path.exists()

        loaded_tensor, loaded_label = torch.load(str(save_path), weights_only=True)
        assert loaded_tensor.shape == (1, H, W)
        assert loaded_label == 1

        expected_slice = torch.from_numpy(volume_data[D // 2]).float()
        assert torch.allclose(loaded_tensor.squeeze(0), expected_slice)


class TestMinMaxNormalize:
    """Validates dynamic range scaling and the zero-division safety guard."""

    def test_standard_tensor_normalized_to_unit_range(self) -> None:
        """After normalization, min must be 0.0 and max must be 1.0."""
        transform = MinMaxNormalize()
        t = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.0])
        result = transform(t)
        assert float(result.min()) == pytest.approx(0.0, abs=1e-6)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-6)

    def test_all_values_in_unit_interval(self) -> None:
        """Every normalized value must lie within [0.0, 1.0]."""
        transform = MinMaxNormalize()
        result = transform(torch.randn(64, 64))
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_constant_tensor_returned_unchanged_without_nan_or_inf(self) -> None:
        """A uniform tensor (Δ < 1e-6) must be returned as-is with no NaN or Inf."""
        transform = MinMaxNormalize()
        t = torch.full((8, 8), 7.5)
        result = transform(t)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert torch.equal(result, t)


class TestOasisDataset:
    """Validates indexing, length, and tensor-type contracts for OasisDataset."""

    def _create_pt_files(self, directory: Path, n: int = 8) -> None:
        """Writes n synthetic (Tensor, int) .pt files to the given directory.

        Args:
            directory: Destination directory for .pt files.
            n: Number of files to create.
        """
        for i in range(n):
            torch.save(
                (torch.randn(1, 32, 32), i % 2), str(directory / f"s_{i:04d}.pt")
            )

    def test_length_equals_number_of_pt_files(self, tmp_path: Path) -> None:
        """__len__ must equal the count of .pt files in the directory."""
        self._create_pt_files(tmp_path, 5)
        assert len(OasisDataset(directory_path=str(tmp_path))) == 5

    def test_getitem_returns_tensor_and_int_label(self, tmp_path: Path) -> None:
        """__getitem__ must return a Tensor for the image and an int for the label."""
        self._create_pt_files(tmp_path, 4)
        img, label = OasisDataset(directory_path=str(tmp_path))[0]
        assert isinstance(img, Tensor)
        assert isinstance(label, int)

    def test_getitem_tensor_shape_is_preserved(self, tmp_path: Path) -> None:
        """The tensor shape (1, 32, 32) stored in .pt must survive the load cycle."""
        self._create_pt_files(tmp_path, 4)
        img, _ = OasisDataset(directory_path=str(tmp_path))[0]
        assert img.shape == (1, 32, 32)


class TestOasisDataLoaderNiftiMode:
    """Validates batch size, shuffle, and missing-path contracts for nifti mode."""

    def _populate_nifti_dir(self, directory: Path, n: int = 8) -> None:
        """Writes n synthetic .pt slice files to mimic a processed NIfTI directory.

        Args:
            directory: Target directory (must already exist).
            n: Number of .pt files to create.
        """
        for i in range(n):
            torch.save(
                (torch.randn(1, 128, 128), i % 2), str(directory / f"s_{i:04d}.pt")
            )

    def test_train_loader_respects_batch_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DataLoader must yield batches of exactly the configured batch size."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        self._populate_nifti_dir(train_dir, 8)
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        loader = OasisDataLoader(batch_size=4).get_data_loader(is_train=True)
        imgs, _ = next(iter(loader))
        assert imgs.shape[0] == 4

    def test_train_loader_uses_random_sampler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training DataLoader must be created with shuffle=True (RandomSampler)."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        self._populate_nifti_dir(train_dir, 8)
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        loader = OasisDataLoader(batch_size=4).get_data_loader(is_train=True)
        assert isinstance(loader.sampler, RandomSampler)

    def test_test_loader_uses_sequential_sampler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test DataLoader must be created with shuffle=False (SequentialSampler)."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        self._populate_nifti_dir(test_dir, 8)
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        loader = OasisDataLoader(batch_size=4).get_data_loader(is_train=False)
        assert isinstance(loader.sampler, SequentialSampler)

    def test_missing_train_directory_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FileNotFoundError must be raised when the nifti train directory is absent."""
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            OasisDataLoader(batch_size=4).get_data_loader(is_train=True)

    def test_empty_train_directory_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FileNotFoundError must be raised when the nifti train directory is empty."""
        (tmp_path / "train").mkdir()
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            OasisDataLoader(batch_size=4).get_data_loader(is_train=True)
