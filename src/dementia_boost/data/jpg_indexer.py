"""Regex patient ID indexer and immutable train/test CSV split generator.

This module provides the `JpgDataIndexer` class to parse patient identifiers
from 2D MRI JPG filenames, enforce subject-level train/test isolation, and
generate immutable index CSV files for downstream PyTorch DataLoaders.
"""

import csv
import os
import random
import re
from collections import defaultdict


class JpgDataIndexer:
    """Parses JPG filenames, isolates patient cohorts, and exports CSV indexes.

    Expects a directory containing class-specific subfolders of 2D MRI JPGs.
    Extracts base patient IDs using regular expressions, groups image paths by
    subject, applies manual overrides, and performs patient-level cohort splits
    to prevent data leakage.

    Attributes:
        raw_jpg_dir: Path to directory containing dataset class subfolders.
        output_dir: Destination directory for generated CSV index files.
    """

    def __init__(self, raw_jpg_dir: str, output_dir: str) -> None:
        """Initializes the indexer and ensures the output directory exists.

        Args:
            raw_jpg_dir: Path to directory containing dataset folders.
            output_dir: Destination folder for the generated index CSVs.
        """
        self.raw_jpg_dir = raw_jpg_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def index_and_split(
        self,
        split_ratio: float = 0.7,
        seed: int = 158,
        manual_train_ids: list[str] | None = None,
        manual_test_ids: list[str] | None = None,
    ) -> None:
        """Groups images by subject ID, applies overrides, and saves CSV indexes.

        Scans `raw_jpg_dir` for images, parses patient IDs, builds a mapping
        from patient to `(filepath, label)`, applies manual cohort assignments,
        and splits remaining subjects deterministically using the specified seed.

        Args:
            split_ratio: Proportion of subjects to allocate to the training
                cohort (between 0.0 and 1.0). Defaults to 0.7.
            seed: Random seed for reproducible subject shuffling. Defaults
                to 158.
            manual_train_ids: Optional list of patient IDs forced into the
                training cohort. Defaults to None.
            manual_test_ids: Optional list of patient IDs forced into the
                test cohort. Defaults to None.
        """
        manual_train: set[str] = {pid.lower() for pid in (manual_train_ids or [])}
        manual_test: set[str] = {pid.lower() for pid in (manual_test_ids or [])}

        patient_dict: dict[str, list[tuple[str, int]]] = defaultdict(list)
        label_map = {"NonDemented": 0, "Demented": 1}

        for class_name, label in label_map.items():
            class_dir = os.path.join(self.raw_jpg_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for file in os.listdir(class_dir):
                if not file.lower().endswith((".jpg", ".jpeg")):
                    continue

                try:
                    patient_id = self._extract_patient_id(file)
                    img_path = os.path.abspath(os.path.join(class_dir, file))
                    patient_dict[patient_id].append((img_path, label))
                except ValueError:
                    continue

        all_subjects = set(patient_dict.keys())

        invalid_train = manual_train - all_subjects
        invalid_test = manual_test - all_subjects
        if invalid_train or invalid_test:
            print(
                f"Warning: Overrides contain unrecognized patient IDs: "
                f"{invalid_train | invalid_test}"
            )

        remaining_subjects = list(all_subjects - manual_train - manual_test)

        random.seed(seed)
        random.shuffle(remaining_subjects)

        target_train_size = int(len(all_subjects) * split_ratio)
        needed_for_train = max(0, target_train_size - len(manual_train))

        final_train_ids = list(manual_train) + remaining_subjects[:needed_for_train]
        final_test_ids = list(manual_test) + remaining_subjects[needed_for_train:]

        self._save_to_csv("train_jpg_index.csv", final_train_ids, patient_dict)
        self._save_to_csv("test_jpg_index.csv", final_test_ids, patient_dict)

        print(
            f"Index successfully generated: "
            f"{len(final_train_ids)} Train, {len(final_test_ids)} Test subjects."
        )

    def _extract_patient_id(self, filename: str) -> str:
        """Extracts the normalized patient identifier from a filename using regex.

        Matches patterns like "oas" or "oas2" followed by an underscore and
        digits (e.g., "Oas_001", "OAS_0004", "Oas2_001").

        Args:
            filename: The name of the JPG image file.

        Returns:
            The extracted patient ID in lowercase.

        Raises:
            ValueError: If no valid patient ID pattern is found in the filename.
        """
        match = re.search(r"(?i)(oas2?_\d+)", filename)
        if match:
            return match.group(1).lower()
        raise ValueError(f"Could not extract Patient ID from: {filename}")

    def _save_to_csv(
        self,
        filename: str,
        ids: list[str],
        patient_dict: dict[str, list[tuple[str, int]]],
    ) -> None:
        """Writes patient image filepaths and labels to an immutable index CSV.

        Args:
            filename: Name of the CSV file to create within `output_dir`.
            ids: List of patient IDs to include in this index.
            patient_dict: Dictionary mapping patient IDs to lists of
                `(filepath, label)` tuples.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])

            for pid in ids:
                for img_path, label in patient_dict[pid]:
                    writer.writerow([img_path, label])
