"""NIfTI 3D to 2D slice ETL pipeline and patient-isolated cohort splitting.

This module provides the `OasisDataProcessor` class to stream raw 3D NIfTI/HDR
brain MRI scans from disk, extract central 2D axial slices, prevent patient-level
data leakage across longitudinal visits, and serialize tensors to `.pt` files.
"""

import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.spatialimages import SpatialImage
from numpy import random


class OasisDataProcessor:
    """Handles the ETL process for raw NIfTI/HDR medical volume images.

    This processor streams files from disk to prevent out-of-memory (OOM)
    errors when processing large MRI collections. It guarantees strict
    patient-level train/test isolation so that multiple longitudinal visits for
    the same subject never cross between cohorts.

    Attributes:
        RAW_PATH: Base directory where raw NIfTI/HDR scans reside. Defaults
            to "./data/raw".
        PROCESSED_PATH: Base directory where processed `.pt` tensor files are
            saved. Defaults to "./data/results".
        csv_path: Path to the metadata CSV file containing OASIS clinical data.
        train_dir: Destination directory for training set `.pt` files.
        test_dir: Destination directory for test set `.pt` files.
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

        self.train_dir = os.path.join(self.PROCESSED_PATH, "train")
        self.test_dir = os.path.join(self.PROCESSED_PATH, "test")
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def process_and_save(
        self,
        split_ratio: float = 0.7,
        manual_train_ids: list[str] | None = None,
        manual_test_ids: list[str] | None = None,
    ) -> None:
        """Executes the complete ETL and patient-split pipeline.

        Parses the metadata CSV, splits unique Subject IDs into train/test
        cohorts respecting manual overrides, streams raw NIfTI files from disk,
        extracts the central 2D axial slice, and serializes processed tensors
        along with labels as `.pt` files.

        Args:
            split_ratio: Target proportion of subjects to allocate to the
                training cohort. Defaults to 0.7 (70%).
            manual_train_ids: Optional list of Subject IDs forced into the
                training set. Defaults to None.
            manual_test_ids: Optional list of Subject IDs forced into the
                testing set. Defaults to None.
        """
        manual_train_ids = manual_train_ids or []
        manual_test_ids = manual_test_ids or []

        subject_metadata = self._parse_csv()

        train_subjects, test_subjects = self._split_subjects(
            subject_metadata,
            split_ratio,
            manual_train_ids,
            manual_test_ids,
        )

        print(
            f"Splitting complete: {len(train_subjects)} Train subjects, ",
            f"{len(test_subjects)} Test subjects.",
        )

        print("Processing Training Data...")
        self._process_subset(train_subjects, subject_metadata, self.train_dir)

        print("Processing Test Data...")
        self._process_subset(test_subjects, subject_metadata, self.test_dir)

    def _parse_csv(self) -> dict[str, int]:
        """Reads metadata CSV and maps subjects to binary dementia labels.

        Excludes subjects marked with the 'Converted' status to preserve clean
        binary classes ('Nondemented' -> 0, 'Demented' -> 1).

        Returns:
            A dictionary mapping each Subject ID to its binary integer label.
        """
        df = pd.read_csv(self.csv_path)

        label_mapping: dict[str, int] = {"Nondemented": 0, "Demented": 1}

        df_filtered = df[df["Group"] != "Converted"]

        return {
            str(subj): label_mapping[str(group)]
            for subj, group in zip(
                df_filtered["Subject ID"],
                df_filtered["Group"],
                strict=False,
            )
            if str(group) in label_mapping
        }

    def _split_subjects(
        self,
        metadata: dict[str, int],
        ratio: float,
        manual_train: list[str],
        manual_test: list[str],
    ) -> tuple[set[str], set[str]]:
        """Splits the dataset strictly by Subject ID to prevent patient leakage.

        Ensures that all longitudinal exams for a given patient remain within
        the same cohort. Respects manual ID overrides and randomly allocates
        the remaining subjects according to the split ratio.

        Args:
            metadata: Mapping of Subject IDs to binary labels.
            ratio: Target proportion of subjects for the training set.
            manual_train: List of Subject IDs manually assigned to training.
            manual_test: List of Subject IDs manually assigned to testing.

        Returns:
            A tuple of two sets containing the Subject IDs for the training
            and testing cohorts, respectively.
        """
        all_subjects = set(metadata.keys())

        train_set = set(manual_train)
        test_set = set(manual_test)

        remaining = list(all_subjects - train_set - test_set)
        random.shuffle(remaining)

        target_train_size = int(len(all_subjects) * ratio)
        needed_for_train = max(0, target_train_size - len(train_set))

        train_set.update(remaining[:needed_for_train])
        test_set.update(remaining[needed_for_train:])

        return train_set, test_set

    def _process_subset(
        self,
        subjects: set[str],
        metadata: dict[str, int],
        output_dir: str,
    ) -> None:
        """Streams, extracts axial slices, and saves tensors for a subject cohort.

        For each subject in the cohort, searches the raw data directory for all
        associated visits and HDR/NIfTI volume files. Loads each volume using
        Nibabel, extracts the central 2D axial slice, adds a channel dimension
        to produce shape [1, H, W], and saves the (tensor, label) tuple as a
        `.pt` file.

        Args:
            subjects: Set of Subject IDs assigned to this cohort.
            metadata: Mapping of Subject IDs to binary integer labels.
            output_dir: Destination directory path for the saved `.pt` files.
        """
        processed_count = 0

        for subject_id in subjects:
            label = metadata[subject_id]

            search_pattern = os.path.join(
                self.RAW_PATH,
                f"{subject_id}_MR*",
                "RAW",
                "*.hdr",
            )
            exam_files = glob.glob(search_pattern)

            for file_path in exam_files:
                try:
                    img_obj = nib.load(file_path)
                    if not isinstance(img_obj, SpatialImage):
                        continue

                    volume_data = np.squeeze(img_obj.get_fdata())

                    if volume_data.ndim != 3:
                        print(
                            f"Skipping {file_path}: unexpected dimensions",
                            f"{volume_data.shape}",
                        )
                        continue

                    middle_idx = volume_data.shape[0] // 2
                    slice_2d = volume_data[middle_idx, :, :]

                    tensor_2d = torch.from_numpy(slice_2d).float()

                    tensor_volume = tensor_2d.unsqueeze(0)

                    parts = file_path.split(os.sep)
                    visit_folder = parts[-3]
                    exam_name = parts[-1].replace(".nifti.hdr", "")

                    save_name = f"{visit_folder}_{exam_name}.pt"
                    save_path = os.path.join(output_dir, save_name)

                    torch.save((tensor_volume, label), save_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

            print(f" -> Saved {processed_count} files to {output_dir}")
