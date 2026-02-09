import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Lock all random number generators to a single seed for full reproducibility.
    
    Args:
        seed: Integer seed value (recommended: 42, 123, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic CUDA operations (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure Python hash seed is fixed
    os.environ['PYTHONHASHSEED'] = str(seed)