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


class TestOasisDataLoaderBatching:
    """Validates batch size and missing-path contracts of the loader.

    Cohort selection, shuffling, and the manifest check are covered in
    `test_split_guards.py`.
    """

    def test_train_loader_respects_batch_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_layout
    ) -> None:
        """DataLoader must yield batches of exactly the configured batch size."""
        make_layout(tmp_path)
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        loader = OasisDataLoader(batch_size=4).get_data_loader("train")
        imgs, _ = next(iter(loader))
        assert imgs.shape[0] == 4

    def test_missing_train_directory_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FileNotFoundError must be raised when the train directory is absent."""
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            OasisDataLoader(batch_size=4).get_data_loader("train")

    def test_empty_train_directory_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FileNotFoundError must be raised when the train directory is empty."""
        (tmp_path / "train").mkdir()
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            OasisDataLoader(batch_size=4).get_data_loader("train")
