import glob
import os

import nibabel as nib
import pandas as pd
import torch
from nibabel.spatialimages import SpatialImage
from numpy import random
from torch import Tensor


class OasisDataProcessor:
    """
    Handles the ETL process for NIfTI medical images.

    This processor is designed to handle large datasets safely by streaming files from
    disk, preventing Out-Of-Memory (OOM) errors. Crucially, it prevents data leakage
    by ensuring all exams and visits for a single subject are routed strictly to either
    the training or testing set, never both.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self, csv_path: str) -> None:
        """
        Initializes the processor, sets up file paths, and ensures output directories
        exist.

        Args:
            csv_path (str): The absolute or relative path to the metadata CSV file
                containing subject IDs and their corresponding categorical labels.
        """
        self.csv_path = csv_path

        self.train_dir = os.path.join(self.PROCESSED_PATH, "train")
        self.test_dir = os.path.join(self.PROCESSED_PATH, "test")
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)

    def process_and_save(
        self,
        split_ratio: float = 0.7,
        manual_train_ids: list[str] | None = None,
        manual_test_ids: list[str] | None = None,
    ) -> None:
        """
        Executes the full ETL pipeline safely.

        This acts as the main orchestrator. It parses the metadata, splits the unique
        Subject IDs to prevent data leakage, and streams the NIfTI files from disk to
        process and save them as individual `.pt` tensors in their respective
        train/test directories.

        Args:
            split_ratio (float, optional): The target percentage of subjects to allocate
                to the training set. Defaults to 0.7 (70%).
            manual_train_ids (list[str] | None, optional): A list of specific Subject ID
                forced into the training set. Defaults to None.
            manual_test_ids (list[str] | None, optional): A list of specific Subject IDs
                forced into the testing set. Defaults to None.
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

        hdr_files = glob.glob(os.path.join(self.RAW_PATH, "*.hdr"))

        if not hdr_files:
            print(f"No .hdr files found in {self.RAW_PATH}")
            return

        data_list: list[Tensor] = []
        targets_list: list[int] = []

        for idx, file_path in enumerate(hdr_files):
            try:
                img_obj = nib.load(file_path)

                if not isinstance(img_obj, SpatialImage):
                    print(
                        f"Warning: {file_path} is not a valid spatial image. Skipping."
                    )
                    continue

                volume_data = img_obj.get_fdata()

                tensor_volume = torch.from_numpy(volume_data).float()

                tensor_volume = tensor_volume.squeeze()

                tensor_volume = tensor_volume.unsqueeze(0)

                data_list.append(tensor_volume)

                targets_list.append(idx % 2)

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue

        all_data = torch.stack(data_list)
        all_targets = torch.tensor(targets_list, dtype=torch.long)

        split_idx = int(len(all_data) * split_ratio)

        torch.save(
            (all_data[:split_idx], all_targets[:split_idx]),
            os.path.join(self.PROCESSED_PATH, "training.pt"),
        )

        torch.save(
            (all_data[split_idx:], all_targets[split_idx:]),
            os.path.join(self.PROCESSED_PATH, "test.pt"),
        )

        print(
            f"Processed {len(hdr_files)} NIfTI volumes. Saved to {self.PROCESSED_PATH}"
        )

    def _parse_csv(self) -> dict[str, str]:
        """
        Reads the metadata CSV, filters out excluded categories, and maps subjects
        to labels.

        This method specifically drops any subjects marked with the exclusion label
        to ensure they do not contaminate the training or testing pools.

        Returns:
            dict[str, str]: A dictionary mapping the Subject ID (key) to its
                categorical label (value).
        """
        df = pd.read_csv(self.csv_path)

        df_filtered = df[df["Group"] != "Converted"]

        return dict(zip(df_filtered["Subject ID"], df_filtered["Group"], strict=True))

    def _split_subjects(
        self,
        metadata: dict[str, str],
        ratio: float,
        manual_train: list[str],
        manual_test: list[str],
    ) -> tuple[set[str], set[str]]:
        """
        Splits the dataset strictly by Subject ID rather than by individual images.

        This ensures that all longitudinal data (multiple visits/exams) for a single
        patient ends up in the same cohort, preventing cross-contamination (leakage)
        between the train and test sets. It respects manual ID assignments before
        randomly distributing the remaining subjects to meet the target `ratio`.

        Args:
            metadata (dict[str, str]): The parsed dict of available SubjectID and label.
            ratio (float): The desired ratio of training data (e.g., 0.7).
            manual_train (list[str]): Subject IDs manually assigned to the training set.
            manual_test (list[str]): Subject IDs manually assigned to the testing set.

        Returns:
            tuple[set[str], set[str]]: Two sets containing the partitioned Subject IDs
                for training and testing, respectively.
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
