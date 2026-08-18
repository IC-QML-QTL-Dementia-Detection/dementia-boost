"""PyTorch Dataset implementations for OASIS-2 MRI tensors and JPG images.

This module defines `OasisDataset` for loading pre-extracted serialized PyTorch
tensors (`.pt`), and `JpgOasisDataset` for loading 2D JPG images dynamically from
disk using PIL.
"""

import glob
import os
from collections.abc import Callable

import torch
from PIL import Image
from torch import Tensor, load
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class OasisDataset(Dataset):
    """PyTorch Dataset for loading pre-processed serialized tensor files.

    Loads `.pt` files containing pre-extracted central 2D axial slices paired
    with integer diagnostic labels from a target directory.

    Attributes:
        directory_path: Path to the directory containing `.pt` files.
        file_list: List of matching `.pt` filepaths in `directory_path`.
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
        self.file_list = glob.glob(os.path.join(directory_path, "*.pt"))
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


class JpgOasisDataset(Dataset):
    """PyTorch Dataset for loading JPG images dynamically via PIL.

    Reads 2D grayscale JPG files from disk on demand, converts them to tensors,
    and returns them paired with float classification targets.

    Attributes:
        samples: List of tuples containing `(image_path, binary_label)`.
        transform: Optional callable transform applied to the PIL image.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform: Callable | None = None,
    ) -> None:
        """Initializes the dataset with image path and label pairs.

        Args:
            samples: A list of tuples containing (image_path, binary_label).
            transform: Optional callable transform to apply to the data.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            The total count of indexed image samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Loads the image from disk, converts to grayscale, and applies transforms.

        Args:
            idx: The integer index of the item to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its float32
            scalar label tensor.
        """
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("L")

        if self.transform:
            tensor_image = self.transform(image)
        else:
            tensor_image = ToTensor()(image)

        return tensor_image, torch.tensor(label, dtype=torch.float32)
