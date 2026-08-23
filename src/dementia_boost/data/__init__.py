"""Data engineering, ETL pipelines, dataset representations, and data loaders.

This module provides data structures and orchestration pipelines for preprocessing,
patient-level dataset splitting, transformation, indexing, batching, and in-memory
feature embedding caching of OASIS-2 MRI data.
"""

from .data_loader import MinMaxNormalize, OasisDataLoader
from .data_processor import OasisDataProcessor
from .dataset import JpgOasisDataset, OasisDataset
from .embedding_cache import CachedEmbeddingDataset, FeatureCacheManager
from .jpg_indexer import JpgDataIndexer

__all__ = [
    "CachedEmbeddingDataset",
    "FeatureCacheManager",
    "JpgDataIndexer",
    "JpgOasisDataset",
    "MinMaxNormalize",
    "OasisDataLoader",
    "OasisDataProcessor",
    "OasisDataset",
]
