"""Tests for the ETL outputs: patient-level leakage and the split manifest.

Each test drives the public `OasisDataProcessor.process_and_save` on a tiny
synthetic raw folder and inspects the cohort directories and manifest it writes.

Regression coverage
-------------------
- Stale files from an earlier ETL run left a subject in both cohorts.
- The split depended on the per-process string hash seed.
- The split depended on the global NumPy RNG state.
- Subjects missing from one source, or unreadable volumes, were dropped
  without a trace instead of being listed with a reason.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from dementia_boost.data import data_processor
from dementia_boost.data.cohort_audit import find_shared_subjects, list_cohort_subjects
from dementia_boost.data.data_processor import OasisDataProcessor
from dementia_boost.data.split import SubjectSplit
from dementia_boost.data.split_manifest import compute_split_id

_N_SUBJECTS = 12
_CSV_HEADER = "Subject ID,MRI ID,Group,Visit\n"


def _write_volume(raw_dir: Path, visit: str, shape: tuple[int, ...], seed: int) -> None:
    """Writes one synthetic NIfTI/HDR volume under `raw_dir/<visit>/RAW`.

    Args:
        raw_dir: Raw data root.
        visit: Visit folder name, for example `OAS2_0001_MR1`.
        shape: Volume shape; anything other than 3D is skipped by the ETL.
        seed: Seed of the generator that fills the volume.
    """
    raw_path = raw_dir / visit / "RAW"
    raw_path.mkdir(parents=True)
    volume = np.random.default_rng(seed).random(shape).astype(np.float32)
    nib.Nifti1Pair(volume, np.eye(4)).to_filename(str(raw_path / "mpr-1.nifti.hdr"))


def _build_raw(
    raw_dir: Path,
    csv_only: tuple[str, ...] = (),
    raw_only: tuple[str, ...] = (),
    bad_volume: tuple[str, ...] = (),
) -> Path:
    """Writes a synthetic raw folder and its CSV with alternating labels.

    Args:
        raw_dir: Directory that becomes the raw data root.
        csv_only: Extra subjects with a CSV row but no raw folder.
        raw_only: Extra subjects with a raw folder but no CSV row.
        bad_volume: Extra subjects whose only volume is 2D, so it is skipped.

    Returns:
        The path of the written metadata CSV.
    """
    rows = []
    for i in range(1, _N_SUBJECTS + 1):
        subject = f"OAS2_{i:04d}"
        group = "Demented" if i % 2 else "Nondemented"
        rows.append(f"{subject},{subject}_MR1,{group},1")
        _write_volume(raw_dir, f"{subject}_MR1", (4, 8, 8), i)
    for subject in csv_only:
        rows.append(f"{subject},{subject}_MR1,Demented,1")
    for subject in raw_only:
        _write_volume(raw_dir, f"{subject}_MR1", (4, 8, 8), 99)
    for subject in bad_volume:
        rows.append(f"{subject},{subject}_MR1,Demented,1")
        _write_volume(raw_dir, f"{subject}_MR1", (8, 8), 98)
    csv_path = raw_dir / "demographics.csv"
    csv_path.write_text(_CSV_HEADER + "\n".join(rows) + "\n")
    return csv_path


def _processor(
    root: Path, monkeypatch: pytest.MonkeyPatch, **raw_options: tuple[str, ...]
) -> OasisDataProcessor:
    """Builds a processor whose raw and processed roots live under `root`.

    Args:
        root: Temporary directory holding `raw/` and `results/`.
        monkeypatch: pytest fixture used to redirect the class path constants.
        **raw_options: Extra raw-folder cases passed to `_build_raw`.

    Returns:
        A processor reading `root/raw` and writing `root/results`.
    """
    raw_dir = root / "raw"
    csv_path = _build_raw(raw_dir, **raw_options)
    monkeypatch.setattr(OasisDataProcessor, "RAW_PATH", str(raw_dir))
    monkeypatch.setattr(OasisDataProcessor, "PROCESSED_PATH", str(root / "results"))
    return OasisDataProcessor(csv_path=str(csv_path))


def _assignment(results_dir: Path) -> dict[str, set[str]]:
    """Reads which subjects sit in each cohort directory.

    Args:
        results_dir: Directory holding the `train/` and `test/` folders.

    Returns:
        Mapping of cohort name to its subjects.
    """
    return {
        name: list_cohort_subjects(str(results_dir / name))
        for name in ("train", "val", "test")
    }


def test_second_run_with_other_seed_leaves_cohorts_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ETL runs with different seeds must not leave a subject in both cohorts."""
    processor = _processor(tmp_path, monkeypatch)
    processor.process_and_save(seed=0)
    first = _assignment(tmp_path / "results")
    processor.process_and_save(seed=4)
    cohorts = _assignment(tmp_path / "results")

    assert first["test"] != cohorts["test"] or first["train"] != cohorts["train"], (
        "the two seeds must assign differently, or this test proves nothing"
    )
    assert find_shared_subjects(cohorts) == {}


def test_split_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    """The same seed must give the same split in processes with different hashes."""
    _build_raw(tmp_path / "raw")
    script = textwrap.dedent(
        """
        import sys
        from dementia_boost.data.data_processor import OasisDataProcessor

        root = sys.argv[1]
        OasisDataProcessor.RAW_PATH = root + "/raw"
        OasisDataProcessor.PROCESSED_PATH = root + "/results_" + sys.argv[2]
        OasisDataProcessor(csv_path=root + "/raw/demographics.csv").process_and_save(
            seed=42
        )
        """
    )
    assignments = []
    for hash_seed in ("1", "2", "3"):
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), hash_seed],
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
            capture_output=True,
        )
        assignments.append(_assignment(tmp_path / f"results_{hash_seed}"))

    assert assignments[0] == assignments[1] == assignments[2]


def test_etl_leaves_global_rng_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ETL must neither read nor reseed the global NumPy RNG."""
    processor = _processor(tmp_path, monkeypatch)
    np.random.seed(12345)
    expected_next = np.random.rand()
    np.random.seed(12345)

    processor.process_and_save(seed=42)

    assert np.random.rand() == expected_next


def _file_set(results_dir: Path) -> set[str]:
    """Lists every cohort file as `<cohort>/<name>`."""
    return {
        f"{path.parent.name}/{path.name}"
        for path in results_dir.glob("*/*")
        if path.parent.name in ("train", "val", "test")
    }


def _manifest(results_dir: Path) -> dict:
    """Loads the split manifest written next to the cohorts."""
    return json.loads((results_dir / "split_manifest.json").read_text())


def test_stale_file_in_other_cohort_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train subject's file planted in `test/` must be gone after the ETL."""
    processor = _processor(tmp_path, monkeypatch)
    results = tmp_path / "results"
    processor.process_and_save(seed=0)
    train_file = next((results / "train").glob("*.pt"))
    planted = results / "test" / train_file.name
    planted.write_bytes(train_file.read_bytes())

    processor.process_and_save(seed=0)

    assert not planted.exists()
    assert find_shared_subjects(_assignment(results)) == {}


def test_foreign_files_and_directories_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only `.pt` files are owned by the ETL; other files stay untouched."""
    processor = _processor(tmp_path, monkeypatch)
    results = tmp_path / "results"
    processor.process_and_save(seed=0)
    note = results / "train" / "notes.txt"
    note.write_text("keep me")

    processor.process_and_save(seed=4)

    assert note.read_text() == "keep me"


def test_rerun_gives_identical_files_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the ETL twice with the same arguments changes nothing."""
    processor = _processor(tmp_path, monkeypatch)
    results = tmp_path / "results"
    processor.process_and_save(seed=3)
    files = _file_set(results)
    manifest_bytes = (results / "split_manifest.json").read_bytes()

    processor.process_and_save(seed=3)

    assert _file_set(results) == files
    assert (results / "split_manifest.json").read_bytes() == manifest_bytes


def test_interrupted_etl_keeps_previous_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while writing must leave the previous files and manifest intact."""
    processor = _processor(tmp_path, monkeypatch)
    results = tmp_path / "results"
    processor.process_and_save(seed=0)
    files = _file_set(results)
    manifest_bytes = (results / "split_manifest.json").read_bytes()

    real_save = torch.save
    calls = {"n": 0}

    def failing_save(*args, **kwargs) -> None:
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("disk full")
        real_save(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(data_processor.torch, "save", failing_save)
        with pytest.raises(OSError, match="disk full"):
            processor.process_and_save(seed=4)

    assert _file_set(results) == files
    assert (results / "split_manifest.json").read_bytes() == manifest_bytes
    assert not (results / ".staging").exists()


def test_split_id_depends_only_on_the_assignment() -> None:
    """The ID is stable for one assignment and changes when a subject moves."""
    base = SubjectSplit(train=["a", "b"], val=["c"], test=["d"])
    same = SubjectSplit(train=["a", "b"], val=["c"], test=["d"])
    moved = SubjectSplit(train=["a"], val=["c"], test=["b", "d"])

    assert compute_split_id(base) == compute_split_id(same)
    assert compute_split_id(base) != compute_split_id(moved)


def test_manifest_records_assignment_exclusions_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest lists every subject, every exclusion with a reason, and settings."""
    processor = _processor(
        tmp_path,
        monkeypatch,
        csv_only=("OAS2_0050",),
        raw_only=("OAS2_0060",),
        bad_volume=("OAS2_0070",),
    )
    results = tmp_path / "results"
    processor.process_and_save(seed=0, manual_train_ids=["OAS2_0001"])

    manifest = _manifest(results)
    cohorts = _assignment(results)

    assert manifest["seed"] == 0
    assert manifest["manual_train_ids"] == ["OAS2_0001"]
    assert manifest["manual_test_ids"] == []
    assert {s: info["cohort"] for s, info in manifest["subjects"].items()} == {
        s: name for name, subjects in cohorts.items() for s in subjects
    }
    assert all(info["n_files"] == 1 for info in manifest["subjects"].values())
    assert set(manifest["excluded"]) == {"OAS2_0050", "OAS2_0060", "OAS2_0070"}
    assert "raw" in manifest["excluded"]["OAS2_0050"].lower()
    assert "csv" in manifest["excluded"]["OAS2_0060"].lower()
    assert "3d" in manifest["excluded"]["OAS2_0070"].lower()
    assert [item["file"] for item in manifest["skipped_files"]] == [
        "OAS2_0070_MR1/RAW/mpr-1.nifti.hdr"
    ]
    for name, info in manifest["cohorts"].items():
        assert info["n_subjects"] == len(cohorts[name])
        assert sum(info["class_balance"].values()) == len(cohorts[name])
    assert len(manifest["split_id"]) == 12
