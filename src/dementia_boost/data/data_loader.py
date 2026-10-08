"""Data loading pipeline and custom normalization transforms for OASIS-2 NIfTI data.

This module provides the `OasisDataLoader` factory class to instantiate PyTorch
`DataLoader` objects for serialized NIfTI slice tensors, applying resizing and
min-max normalization.
"""

import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from dementia_boost.data.dataset import OasisDataset
from dementia_boost.data.split import COHORT_NAMES
from dementia_boost.data.split_guards import verify_split_layout


class OasisDataLoader:
    """Data loader factory for the processed NIfTI slice tensors.

    Loads pre-processed serialized tensors (`.pt`) written by
    `OasisDataProcessor` and applies the resize and normalization transforms.

    Attributes:
        RESULTS_PATH: Base directory where processed data is stored.
            Defaults to "./data/results".
    """

    RESULTS_PATH = "./data/results"

    def __init__(self, batch_size: int = 4) -> None:
        """Initializes the loader with the specified batch size.

        Args:
            batch_size: The number of samples per batch to load. Defaults to 4.
        """
        self._batch_size = batch_size

    def get_data_loader(self, cohort: str = "train") -> DataLoader:
        """Creates and returns a PyTorch DataLoader for the specified cohort.

        Loads pre-processed `.pt` tensor files from `RESULTS_PATH/<cohort>/`.
        Before returning, it verifies the whole layout against
        `split_manifest.json`, so a stale or leaky directory is never loaded.

        Args:
            cohort: One of "train", "val", or "test". Only the training cohort
                is shuffled. Defaults to "train".

        Returns:
            A PyTorch DataLoader configured with the dataset, batch size,
            shuffling, and memory pinning.

        Raises:
            ValueError: If `cohort` is not one of the three cohort names.
            FileNotFoundError: If the cohort directory is missing or empty, or
                the split manifest does not exist.
            SplitIntegrityError: If the cohort files do not match the manifest
                or two cohorts share a subject.
        """
        if cohort not in COHORT_NAMES:
            raise ValueError(
                f"Unknown cohort {cohort!r}; expected one of {COHORT_NAMES}"
            )
        is_train = cohort == "train"
        use_pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()

        dir_path = os.path.join(self.RESULTS_PATH, cohort)

        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            raise FileNotFoundError(
                f"Processed NIfTI data not found at {dir_path}. "
                "Run OasisDataProcessor().process_and_save() first."
            )
        verify_split_layout(self.RESULTS_PATH)

        dataset = OasisDataset(
            directory_path=dir_path,
            transform=self._get_nifti_transform(),
        )

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=is_train,
            pin_memory=use_pin_memory,
        )

    def _get_nifti_transform(self) -> transforms.Compose:
        """Defines the transformation pipeline for raw NIfTI float tensors.

        The pipeline consists of:
        1. Resizing to 128x128 pixels (with antialiasing enabled).
        2. Min-max normalization to [0.0, 1.0] (per-image).
        3. Standard normalization with mean=0.5 and std=0.5 to map to [-1.0, 1.0].

        Returns:
            A composed torchvision transform pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize((128, 128), antialias=True),
                MinMaxNormalize(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )


class MinMaxNormalize:
    """Transform that scales medical image tensors dynamically to [0.0, 1.0].

    This transform operates on a per-image basis (per-sample scaling) to preserve
    local tissue contrast. If the dynamic range is negligible (< 1e-6), the
    input tensor is returned unchanged to prevent division by zero.
    """

    def __call__(self, tensor: Tensor) -> Tensor:
        """Applies min-max scaling to the input tensor.

        Args:
            tensor: Input image tensor of arbitrary dimensions.

        Returns:
            Scaled tensor with values normalized to [0.0, 1.0] if dynamic range
            permits; otherwise, the original tensor.
        """
        min_val = tensor.min()
        max_val = tensor.max()

        if max_val - min_val > 1e-6:
            return (tensor - min_val) / (max_val - min_val)
        return tensor
