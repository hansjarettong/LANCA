import torch
import random
import numpy as np
import os

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"--- Random Seed Set to {seed} ---")

def count_parameters(model):
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)