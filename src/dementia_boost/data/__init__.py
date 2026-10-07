"""Data engineering, ETL pipeline, dataset representation, and data loaders.

This module provides data structures and orchestration pipelines for preprocessing,
patient-level dataset splitting, transformation, batching, and in-memory feature
embedding caching of OASIS-2 NIfTI MRI data.
"""

from .data_loader import MinMaxNormalize, OasisDataLoader
from .data_processor import OasisDataProcessor
from .dataset import OasisDataset
from .embedding_cache import CachedEmbeddingDataset, FeatureCacheManager

__all__ = [
    "CachedEmbeddingDataset",
    "FeatureCacheManager",
    "MinMaxNormalize",
    "OasisDataLoader",
    "OasisDataProcessor",
    "OasisDataset",
]
