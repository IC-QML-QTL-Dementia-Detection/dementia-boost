import os

from torch.utils.data import DataLoader
from torchvision import transforms

from dementia_boost.data.dataset import OasisDataset


class OasisDataLoader:
    """
    Manages the creation of PyTorch DataLoaders for the processed OASIS dataset.
    Applies resizing and normalization transforms.
    """

    PROCESSED_PATH = "./data/results"

    def __init__(self, batch_size: int = 4) -> None:
        """
        Initializes the loader with specified batch size.

        Args:
            `batch_size`: The number of samples per batch to load.
        """
        self._batch_size = batch_size

    def get_data_loader(self, is_train: bool = True) -> DataLoader:
        """
        Creates and returns a DataLoader for the specified subset.

        Args:
            `is_train`: If True, loads the training set; otherwise, loads the test set.

        Returns:
            A PyTorch DataLoader configured with the dataset and transforms.

        Raises:
            `FileNotFoundError`: If the processed data files do not exist.
        """

        dir_name = "train" if is_train else "test"
        dir_path = os.path.join(self.PROCESSED_PATH, dir_name)

        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            raise FileNotFoundError(
                f"Processed data not found at {dir_path}. "
                "Run OasisDataProcessor().process_and_save() first."
            )

        dataset = OasisDataset(
            directory_path=dir_path,
            transform=self._get_transform(),
        )

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=is_train,
            pin_memory=True,
        )

    def _get_transform(self) -> transforms.Compose:
        """
        Defines the sequence of image transformations for 32-bit float tensors.

        Returns:
            A composed torchvision transform.
        """
        return transforms.Compose(
            [
                transforms.Resize((128, 128), antialias=True),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
