"""NIfTI 3D to 2D slice ETL pipeline and patient-isolated cohort splitting.

This module provides the `OasisDataProcessor` class to stream raw 3D NIfTI/HDR
brain MRI scans from disk, extract central 2D axial slices, prevent patient-level
data leakage across longitudinal visits, and serialize tensors to `.pt` files.
"""

import glob
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.spatialimages import SpatialImage

from dementia_boost.data.split import COHORT_NAMES, SubjectSplit, split_subjects
from dementia_boost.data.split_guards import verify_split_layout
from dementia_boost.data.split_manifest import (
    MANIFEST_NAME,
    build_manifest,
    write_manifest,
)
from dementia_boost.data.subject_ids import (
    build_subject_labels,
    subject_of_visit,
    validate_overrides,
)

STAGING_NAME = ".staging"


class OasisDataProcessor:
    """Handles the ETL process for raw NIfTI/HDR medical volume images.

    This processor streams files from disk to prevent out-of-memory (OOM)
    errors when processing large MRI collections. It guarantees strict
    patient-level isolation so that multiple longitudinal visits for the same
    subject never cross between cohorts.

    Each run builds the whole split in a staging directory and publishes it
    only when every file was written, so an interrupted run leaves the previous
    split untouched. Publishing replaces the `.pt` files of `train/`, `val/`,
    and `test/` and then writes `split_manifest.json`; other files in those
    directories are never touched.

    Attributes:
        RAW_PATH: Base directory where raw NIfTI/HDR scans reside. Defaults
            to "./data/raw".
        PROCESSED_PATH: Base directory where processed `.pt` tensor files are
            saved. Defaults to "./data/results".
        csv_path: Path to the metadata CSV file containing OASIS clinical data.
        train_dir: Destination directory for training set `.pt` files.
        val_dir: Destination directory for validation set `.pt` files.
        test_dir: Destination directory for test set `.pt` files.
        manifest_path: Destination of `split_manifest.json`.
        excluded: Subjects left out of the last run, with the reason.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self, csv_path: str) -> None:
        """Initializes the processor and creates necessary directories.

        Args:
            csv_path: The absolute or relative path to the metadata CSV file
                containing subject IDs and their corresponding diagnostic groups.
        """
        self.csv_path = csv_path
        self.excluded: dict[str, str] = {}

        self.train_dir = os.path.join(self.PROCESSED_PATH, "train")
        self.val_dir = os.path.join(self.PROCESSED_PATH, "val")
        self.test_dir = os.path.join(self.PROCESSED_PATH, "test")
        self.manifest_path = os.path.join(self.PROCESSED_PATH, MANIFEST_NAME)
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)
        for directory in (self.train_dir, self.val_dir, self.test_dir):
            os.makedirs(directory, exist_ok=True)

    def process_and_save(
        self,
        test_ratio: float = 0.3,
        val_ratio: float = 0.2,
        seed: int = 42,
        manual_train_ids: list[str] | None = None,
        manual_test_ids: list[str] | None = None,
    ) -> None:
        """Executes the complete ETL and patient-split pipeline.

        Parses the metadata CSV, scans the raw folder, excludes subjects that
        cannot be used (with a reason), splits the remaining Subject IDs into
        train, val, and test cohorts with `split_subjects`, extracts the central
        2D axial slice of every volume into a staging directory, publishes the
        staged files, writes the split manifest, and verifies the written
        layout against it.

        Args:
            test_ratio: Share of all subjects placed in the test cohort.
                Defaults to 0.3 (30%).
            val_ratio: Share of the non-test subjects placed in the validation
                cohort. Defaults to 0.2 (20%).
            seed: Random seed for deterministic subject cohort partitioning.
                Defaults to 42.
            manual_train_ids: Optional list of Subject IDs forced into the
                training set. Defaults to None.
            manual_test_ids: Optional list of Subject IDs forced into the
                testing set. Defaults to None.

        Raises:
            ValueError: If an override is malformed, unknown (or excluded), or
                in both lists, or if a cohort would lack one of the classes.
        """
        labels = self._parse_csv()
        files_by_subject, seen_subjects, skipped_files = self._scan_raw()
        eligible = self._select_eligible(labels, files_by_subject, seen_subjects)

        manual_train_ids, manual_test_ids = validate_overrides(
            manual_train_ids or [], manual_test_ids or [], eligible
        )

        split = split_subjects(
            eligible,
            test_ratio,
            val_ratio,
            seed,
            manual_train_ids,
            manual_test_ids,
        )

        print(
            f"Splitting complete: {len(split.train)} Train, {len(split.val)} Val, "
            f"{len(split.test)} Test subjects, {len(self.excluded)} excluded."
        )

        staging_root = os.path.join(self.PROCESSED_PATH, STAGING_NAME)
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            self._stage_split(split, eligible, files_by_subject, staging_root)
            self._publish(staging_root)
            write_manifest(
                self.manifest_path,
                build_manifest(
                    split=split,
                    labels=eligible,
                    file_counts={s: len(f) for s, f in files_by_subject.items()},
                    excluded=self.excluded,
                    skipped_files=skipped_files,
                    seed=seed,
                    test_ratio=test_ratio,
                    val_ratio=val_ratio,
                    manual_train_ids=manual_train_ids,
                    manual_test_ids=manual_test_ids,
                ),
            )
            verify_split_layout(self.PROCESSED_PATH)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _parse_csv(self) -> dict[str, int]:
        """Reads metadata CSV and maps subjects to binary dementia labels.

        Subject IDs and group names are canonicalised. Subjects marked
        'Converted' are excluded to preserve clean binary classes
        ('Nondemented' -> 0, 'Demented' -> 1) and are kept, with a reason, in
        `self.excluded`.

        Returns:
            A dictionary mapping each canonical Subject ID to its binary label.

        Raises:
            ValueError: If an ID or group is malformed, or a subject has
                conflicting groups.
        """
        labels, self.excluded = build_subject_labels(pd.read_csv(self.csv_path))
        return labels

    def _scan_raw(
        self,
    ) -> tuple[dict[str, list[str]], set[str], list[dict[str, str]]]:
        """Lists the usable raw volumes per subject without loading voxel data.

        A volume is usable when its header loads and describes a 3D image (size
        1 dimensions are ignored). Anything else is reported in the skipped
        list with a reason.

        Returns:
            A tuple of: usable `.hdr` paths per canonical subject (sorted), the
            subjects that have at least one `.hdr` file whether usable or not,
            and the skipped files as `{"file": relative path, "reason": text}`.
        """
        pattern = os.path.join(self.RAW_PATH, "*_MR*", "RAW", "*.hdr")
        files_by_subject: dict[str, list[str]] = {}
        seen_subjects: set[str] = set()
        skipped: list[dict[str, str]] = []

        for file_path in sorted(glob.glob(pattern)):
            relative = os.path.relpath(file_path, self.RAW_PATH).replace(os.sep, "/")
            try:
                subject = subject_of_visit(file_path.split(os.sep)[-3])
            except ValueError:
                skipped.append(
                    {"file": relative, "reason": "Folder name is not a visit ID"}
                )
                continue
            seen_subjects.add(subject)

            try:
                header = nib.load(file_path)
            except Exception as error:
                skipped.append({"file": relative, "reason": f"Unreadable: {error}"})
                continue
            if not isinstance(header, SpatialImage):
                skipped.append({"file": relative, "reason": "Not a spatial image"})
                continue
            shape = tuple(dim for dim in header.shape if dim != 1)
            if len(shape) != 3:
                skipped.append(
                    {"file": relative, "reason": f"Volume is not 3D, shape {shape}"}
                )
                continue
            files_by_subject.setdefault(subject, []).append(file_path)

        return files_by_subject, seen_subjects, skipped

    def _select_eligible(
        self,
        labels: dict[str, int],
        files_by_subject: dict[str, list[str]],
        seen_subjects: set[str],
    ) -> dict[str, int]:
        """Keeps the labelled subjects that have usable volumes.

        Every other subject is added to `self.excluded` with a reason, so
        nothing is dropped silently.

        Args:
            labels: Canonical subject ID to binary label, from the CSV.
            files_by_subject: Usable volumes per subject, from the raw scan.
            seen_subjects: Subjects with any `.hdr` file in the raw folder.

        Returns:
            The labelled subjects with at least one usable volume.
        """
        for subject in sorted(labels):
            if subject in files_by_subject:
                continue
            self.excluded[subject] = (
                "No readable 3D volume in the raw folder"
                if subject in seen_subjects
                else "No raw data in the raw folder"
            )
        for subject in sorted(seen_subjects):
            if subject not in labels and subject not in self.excluded:
                self.excluded[subject] = "No row in the metadata CSV"
        return {s: label for s, label in labels.items() if s in files_by_subject}

    def _stage_split(
        self,
        split: SubjectSplit,
        labels: dict[str, int],
        files_by_subject: dict[str, list[str]],
        staging_root: str,
    ) -> None:
        """Writes every cohort's tensors into the staging directory.

        Args:
            split: The three cohorts.
            labels: Canonical subject ID to binary label.
            files_by_subject: Usable volumes per subject.
            staging_root: Directory that receives one sub-directory per cohort.

        Raises:
            Exception: Any error from reading or saving a volume. A volume that
                passed the header scan but cannot be read stops the ETL, so a
                subject is never published with missing exams.
        """
        for name, subjects in (
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ):
            output_dir = os.path.join(staging_root, name)
            os.makedirs(output_dir)
            print(f"Processing {name} data...")
            for subject in subjects:
                for file_path in files_by_subject[subject]:
                    self._save_slice(file_path, labels[subject], output_dir)

    def _save_slice(self, file_path: str, label: int, output_dir: str) -> None:
        """Extracts the central axial slice of one volume and saves it.

        Loads the volume with Nibabel, takes the middle slice along the first
        axis, adds a channel dimension to produce shape [1, H, W], and saves
        the (tensor, label) tuple as `<visit folder>_<exam name>.pt`.

        Args:
            file_path: Path of the `.hdr` file.
            label: Binary label of the subject.
            output_dir: Destination directory of the `.pt` file.
        """
        image = nib.load(file_path)
        if not isinstance(image, SpatialImage):
            raise TypeError(f"Not a spatial image: {file_path}")
        volume = np.squeeze(image.get_fdata())
        slice_2d = volume[volume.shape[0] // 2, :, :]
        tensor_volume = torch.from_numpy(slice_2d).float().unsqueeze(0)

        parts = file_path.split(os.sep)
        visit_folder = parts[-3]
        exam_name = parts[-1].replace(".nifti.hdr", "")
        torch.save(
            (tensor_volume, label),
            os.path.join(output_dir, f"{visit_folder}_{exam_name}.pt"),
        )

    def _publish(self, staging_root: str) -> None:
        """Replaces the `.pt` files of the cohort directories with the staged ones.

        Only `.pt` files are removed; the directories and any other file in
        them stay.

        Args:
            staging_root: Directory holding the staged `train`, `val`, `test`.
        """
        final_dirs = {
            "train": self.train_dir,
            "val": self.val_dir,
            "test": self.test_dir,
        }
        for name in COHORT_NAMES:
            for stale in glob.glob(os.path.join(final_dirs[name], "*.pt")):
                os.remove(stale)
            staged_dir = os.path.join(staging_root, name)
            for staged in sorted(os.listdir(staged_dir)):
                os.replace(
                    os.path.join(staged_dir, staged),
                    os.path.join(final_dirs[name], staged),
                )
