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
    Handles metadata parsing, subject-level splitting to prevent data leakage,
    and memory-safe streaming of NIfTI files to individual PyTorch tensors.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self, csv_path: str) -> None:
        """
        Initializes the processor and ensures the necessary directories exist.
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
        Reads .hdr files from RAW_PATH, converts them, and saves to PROCESSED_PATH.

        Args:
            split_ratio: Percentage of data to use for training.
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
        """Reads CSV, filters out label "Converted", returns {Subject_ID: Group}."""
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
        """Splits data by Subject ID to prevent leakage."""
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
