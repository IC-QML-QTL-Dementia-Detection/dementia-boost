import glob
import os

import cv2
import torch
from torch import Tensor


class OasisDataProcessor:
    """
    Handles the ETL process: loads raw .hdr images, converts them to
    PyTorch float32 tensors, and saves them to .pt files for fast loading.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self) -> None:
        """
        Initializes the processor and ensures the necessary directories exist.
        """
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)

    pass

    # TODO: implement the same split logic as the paper
    def process_and_save(self, split_ratio: float) -> None:
        """
        Reads .hdr files from RAW_PATH, converts them, and saves to PROCESSED_PATH.

        Args:
            split_ratio: Percentage of data to use for training.
        """
        hdr_files = glob.glob(os.path.join(self.RAW_PATH, "*.hdr"))

        if not hdr_files:
            print(f"No .hdr files found in {self.RAW_PATH}")
            return

        data_list: list[Tensor] = []
        targets_list: list[int] = []

        for idx, file_path in enumerate(hdr_files):
            img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to load {file_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()

            data_list.append(tensor_img)

            targets_list.append(idx % 2)

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

        print(f"Processed {len(hdr_files)} files. Saved to {self.PROCESSED_PATH}")
