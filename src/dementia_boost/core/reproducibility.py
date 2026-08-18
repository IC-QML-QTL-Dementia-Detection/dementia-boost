"""Reproducibility utilities for deterministic experiment execution.

This module provides functions to lock pseudo-random number generator (RNG)
seeds across Python's built-in `random`, NumPy, and PyTorch (CPU and CUDA),
as well as configuring cuDNN backend flags for strict determinism.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Locks the random seed across all underlying libraries.

    Enforces deterministic and reproducible training and evaluation runs by
    seeding Python's `random`, NumPy's RNG, and PyTorch's CPU/CUDA RNGs.
    Additionally, sets `torch.backends.cudnn.deterministic` to True and
    `torch.backends.cudnn.benchmark` to False.

    Args:
        seed: The master seed value. Defaults to 42.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
