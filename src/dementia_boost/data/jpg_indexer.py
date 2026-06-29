import csv
import os
import random
import re
from collections import defaultdict


class JpgDataIndexer:
    """
    Parses JPG filenames to extract Patient IDs, splits the dataset
    by patient to prevent leakage, and saves an immutable CSV index.

    The indexer expects a directory structure where each class is a subfolder
    containing JPG files. It extracts patient IDs using a regex pattern, groups
    files by patient, and creates train/test splits at the patient level to
    avoid cross‑contamination.

    Attributes:
        raw_jpg_dir (str): Path to the directory containing class subfolders.
        output_dir (str): Destination directory for the generated CSV index files.
    """

    def __init__(self, raw_jpg_dir: str, output_dir: str) -> None:
        """
        Initializes the indexer and ensures the output directory exists.

        Args:
            raw_jpg_dir (str): Path to directory containing dataset folders.
                The folder is expected to contain subfolders named after classes.
            output_dir (str): Destination folder for the generated index CSVs.
                The directory is created if it does not exist.
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
        """
        Groups images dynamically by extracted subject ID, enforces manual cohort
        overrides, and saves separate deterministic indexing sets.

        The method scans the raw_jpg_dir for images, extracts patient IDs,
        and builds a dictionary mapping each patient to a list of (filepath, label).
        It then applies manual overrides (if provided) and splits the remaining
        subjects randomly (with a fixed seed) according to the split ratio.

        Args:
            split_ratio (float): Proportion of subjects to allocate to the training set.
                Must be between 0 and 1. Defaults to 0.7.
            seed (int): Random seed for reproducible shuffling. Defaults to 158.
            manual_train_ids (list[str] | None): List of patient IDs to force into
                the training set. Case‑insensitive. Defaults to None.
            manual_test_ids (list[str] | None): List of patient IDs to force into
                the test set. Case‑insensitive. Defaults to None.

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
        """
        Extracts the base patient ID using Regex to handle naming inconsistencies.

        The method looks for a pattern matching "oas" or "oas2" followed by an
        underscore and digits (e.g., "Oas_001", "OAS_0004", "Oas2_001"). It is
        case‑insensitive and returns the matched ID in lowercase.

        Args:
            filename (str): The name of the JPG file (e.g., "Oas_001 (1).jpg").

        Returns:
            str: The extracted patient ID in lowercase (e.g., "oas_001").

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
        """
        Writes the index for a given list of patient IDs to a CSV file.

        The CSV contains two columns: "filepath" (absolute path) and "label"
        (0 for NonDemented, 1 for Demented). Each image belonging to the listed
        patients is written as a separate row.

        Args:
            filename (str): Name of the CSV file to create.
            ids (list[str]): List of patient IDs to include in this index.
            patient_dict (dict[str, list[tuple[str, int]]]): Dictionary mapping
                patient IDs to lists of (filepath, label) tuples.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])

            for pid in ids:
                for img_path, label in patient_dict[pid]:
                    writer.writerow([img_path, label])
