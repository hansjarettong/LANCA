import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class BaseCausalModel(nn.Module, ABC):
    """
    Abstract base class to ensure all models (LANCA, BetaVAE, CausalVAE)
    adhere to the standard Trainer interface.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Must return:
        {
            "x_recon_logits": Tensor (B, C, H, W) - RAW LOGITS
            "z": Tensor (B, latent_dim) - Representation for downstream tasks
        }
        """
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, fwd_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Must return:
        {
            "total": Tensor (scalar),
            "recon": Tensor (scalar),
            ...
        }
        """
        pass