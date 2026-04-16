import glob
import os
from collections.abc import Callable

from torch import Tensor, load
from torch.utils.data import Dataset


class OasisDataset(Dataset):
    """
    Custom PyTorch Dataset to load pre-processed tensor files.
    """

    def __init__(
        self,
        directory_path: str,
        transform: Callable | None = None,
    ) -> None:
        """
        Args:
            `directory_path`: Path to the 'train' or 'test' directory containing
                .pt files.
            `transform`: Optional callable transform to apply to the data.
        """
        self.directory_path = directory_path
        self.file_list = glob.glob(os.path.join(directory_path, "*.pt"))
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Retrieves the image and label at the specified index.

        Args:
            `idx`: The index of the item to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its label.
        """
        file_path = self.file_list[idx]

        img, target = load(file_path, weights_only=True)

        if self.transform:
            img = self.transform(img)

        return img, target
