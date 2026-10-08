"""PyTorch Dataset implementation for OASIS-2 MRI tensors.

This module defines `OasisDataset` for loading pre-extracted serialized PyTorch
tensors (`.pt`).
"""

import glob
import os
from collections.abc import Callable

from torch import Tensor, load
from torch.utils.data import Dataset


class OasisDataset(Dataset):
    """PyTorch Dataset for loading pre-processed serialized tensor files.

    Loads `.pt` files containing pre-extracted central 2D axial slices paired
    with integer diagnostic labels from a target directory.

    Attributes:
        directory_path: Path to the directory containing `.pt` files.
        file_list: Sorted list of matching `.pt` filepaths in `directory_path`,
            so sample order does not depend on the filesystem.
        transform: Optional callable transform applied to the loaded tensor.
    """

    def __init__(
        self,
        directory_path: str,
        transform: Callable | None = None,
    ) -> None:
        """Initializes the dataset from a directory of tensor files.

        Args:
            directory_path: Path to the directory containing `.pt` files.
            transform: Optional callable transform to apply to the data.
        """
        self.directory_path = directory_path
        self.file_list = sorted(glob.glob(os.path.join(directory_path, "*.pt")))
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            The number of `.pt` files discovered in `directory_path`.
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Retrieves the image tensor and label at the specified index.

        Args:
            idx: The integer index of the item to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its integer
            label.
        """
        file_path = self.file_list[idx]

        img, target = load(file_path, weights_only=True)

        if self.transform:
            img = self.transform(img)

        return img, target
