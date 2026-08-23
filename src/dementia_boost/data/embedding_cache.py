"""In-memory dataset representations and feature caching utilities.

This module provides `CachedEmbeddingDataset` and `FeatureCacheManager` to
extract, serialize, and stream pre-computed backbone embeddings for fast,
I/O-free transfer learning training loops.
"""

import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class CachedEmbeddingDataset(Dataset):
    """In-memory PyTorch Dataset storing pre-extracted feature embeddings.

    Holds pre-computed backbone representations paired with target labels in
    contiguous tensors, eliminating redundant forward passes through frozen
    convolutional layers and bypassing disk I/O during training.

    Attributes:
        features: Tensor of extracted feature vectors of shape
            `(num_samples, in_features)`.
        labels: Tensor of target diagnostic labels of shape
            `(num_samples, 1)`.
    """

    def __init__(self, features: Tensor, labels: Tensor) -> None:
        """Initializes the cached embedding dataset.

        Args:
            features: Tensor containing feature embeddings of shape
                `(num_samples, in_features)`.
            labels: Tensor containing binary labels of shape
                `(num_samples, 1)` or `(num_samples,)`.
        """
        self.features = features.float()
        self.labels = labels.float().view(-1, 1)

    def __len__(self) -> int:
        """Returns the total number of cached feature samples.

        Returns:
            The integer count of samples in the dataset.
        """
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Retrieves the feature embedding and label at the specified index.

        Args:
            idx: The integer index of the item to retrieve.

        Returns:
            A tuple containing the feature vector tensor and label tensor.
        """
        return self.features[idx], self.labels[idx]


class FeatureCacheManager:
    """Manager for extracting, serializing, and batching feature embeddings.

    Provides static utility methods to extract spatial feature maps from a
    convolutional backbone, serialize representations to disk, and instantiate
    high-throughput in-memory PyTorch DataLoaders.

    Attributes:
        DEFAULT_BATCH_SIZE: Default batch size for cached DataLoader (64).
    """

    DEFAULT_BATCH_SIZE: int = 64

    @staticmethod
    def extract_features(
        feature_extractor: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Extracts and concatenates feature embeddings from a DataLoader.

        Passes all image batches through the feature extractor in evaluation
        mode without gradient computation, flattens spatial dimensions, and
        compiles representations into contiguous CPU tensors.

        Args:
            feature_extractor: Convolutional backbone network.
            data_loader: PyTorch DataLoader providing input image batches.
            device: Hardware accelerator device used for extraction.

        Returns:
            A tuple containing:
                - `all_features`: Extracted representations tensor of shape
                  `(num_samples, flattened_features)`.
                - `all_labels`: Target labels tensor of shape
                  `(num_samples, 1)`.
        """
        feature_extractor.eval()
        feature_extractor.to(device)

        collected_features: list[Tensor] = []
        collected_labels: list[Tensor] = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.float().view(-1, 1)

                features = feature_extractor(images)
                flattened = torch.flatten(features, start_dim=1)

                collected_features.append(flattened.cpu())
                collected_labels.append(labels.cpu())

        all_features = torch.cat(collected_features, dim=0)
        all_labels = torch.cat(collected_labels, dim=0)

        return all_features, all_labels

    @staticmethod
    def create_cached_loader(
        features: Tensor,
        labels: Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
    ) -> DataLoader:
        """Instantiates a PyTorch DataLoader over cached in-memory embeddings.

        Args:
            features: Tensor containing feature embeddings of shape
                `(num_samples, in_features)`.
            labels: Tensor containing binary labels of shape
                `(num_samples, 1)`.
            batch_size: Number of embedding samples per batch. Defaults to 64.
            shuffle: Whether to shuffle samples at each epoch. Defaults to True.

        Returns:
            A PyTorch DataLoader configured with `CachedEmbeddingDataset`.
        """
        dataset = CachedEmbeddingDataset(features=features, labels=labels)
        use_pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=use_pin_memory,
        )

    @staticmethod
    def save_cache(
        features: Tensor,
        labels: Tensor,
        save_path: str,
    ) -> None:
        """Serializes extracted feature embeddings and labels to disk.

        Args:
            features: Tensor containing feature embeddings.
            labels: Tensor containing target labels.
            save_path: Filepath where the `.pt` archive will be written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {"features": features, "labels": labels},
            save_path,
        )

    @staticmethod
    def load_cache(cache_path: str) -> tuple[Tensor, Tensor]:
        """Loads serialized feature embeddings and labels from disk.

        Args:
            cache_path: Filepath to the serialized `.pt` cache archive.

        Returns:
            A tuple containing the loaded `features` and `labels` tensors.

        Raises:
            FileNotFoundError: If `cache_path` does not exist on disk.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cached embedding file not found: {cache_path}")

        payload = torch.load(cache_path, weights_only=True)
        return payload["features"], payload["labels"]
