import glob
import os

import nibabel as nib
import torch
from nibabel.spatialimages import SpatialImage
from torch import Tensor


class OasisDataProcessor:
    """
    Handles the ETL process: loads raw NIfTI .hdr/.img pairs, converts them to
    3D PyTorch tensors, and saves them to .pt files.
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
            try:
                img_obj = nib.load(file_path)

                if not isinstance(img_obj, SpatialImage):
                    print(
                        f"Warning: {file_path} is not a valid spatial image. Skipping."
                    )
                    continue

                volume_data = img_obj.get_fdata()

                tensor_volume = torch.from_numpy(volume_data).float()

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
