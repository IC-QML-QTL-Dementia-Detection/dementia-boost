"""Unit tests for the NIfTI data engineering and patient isolation pipeline.

This module validates:
- ``OasisDataProcessor`` CSV parsing (the slice extraction is checked through
  the ETL in `test_split_leak.py`).
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

    def test_mixed_labels_parsed_correctly(self, tmp_path: Path) -> None:
        """Asserts that a mixed CSV maps Nondemented to 0 and Demented to 1, and
        leaves out the Converted subject."""
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


class TestMinMaxNormalize:
    """Validates dynamic range scaling and the zero-division safety guard."""

    def test_standard_tensor_normalized_to_unit_range(self) -> None:
        """After normalization, min must be 0.0 and max must be 1.0."""
        transform = MinMaxNormalize()
        t = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.0])
        result = transform(t)
        assert float(result.min()) == pytest.approx(0.0, abs=1e-6)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-6)

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

    def test_length_types_and_shape(self, tmp_path: Path) -> None:
        """__len__ equals the count of .pt files, and __getitem__ returns the
        stored (1, 32, 32) Tensor with an int label."""
        self._create_pt_files(tmp_path, 5)
        dataset = OasisDataset(directory_path=str(tmp_path))

        img, label = dataset[0]

        assert len(dataset) == 5
        assert isinstance(img, Tensor)
        assert img.shape == (1, 32, 32)
        assert isinstance(label, int)


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

    def test_subject_ids_follow_the_order_the_loader_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_layout
    ) -> None:
        """The subject of each slice is listed in the order the evaluation
        loader yields the slices: same length, and each slice's label is its
        subject's label (subject number modulo 2 in the fixture)."""
        make_layout(tmp_path)
        monkeypatch.setattr(OasisDataLoader, "RESULTS_PATH", str(tmp_path))
        loader_manager = OasisDataLoader(batch_size=3)

        labels = [
            int(label)
            for _, batch in loader_manager.get_data_loader("test")
            for label in batch
        ]
        subjects = loader_manager.get_subject_ids("test")

        assert len(subjects) == len(labels)
        assert [int(subject[-4:]) % 2 for subject in subjects] == labels

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
