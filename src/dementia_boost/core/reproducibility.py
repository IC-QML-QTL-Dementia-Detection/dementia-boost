import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Locks the random seed across all underlying libraries to ensure
    deterministic and reproducible training runs.

    Args:
        seed (int): The master seed value. Defaults to 42.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
