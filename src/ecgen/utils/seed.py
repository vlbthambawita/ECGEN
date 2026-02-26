import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (if available).
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not installed; skip torch-specific seeding.
        return
