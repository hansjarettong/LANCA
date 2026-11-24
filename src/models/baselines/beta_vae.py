import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

from .base_model import BaseCausalModel
from ..backbones import SharedEncoder, SharedDecoder

@dataclass
class BetaVAEConfig:
    img_channels: int = 3
    img_size: int = 96
    latent_dim: int = 64  # CNN bottleneck
    z_dim: int = 10       # VAE latent size
    
    # Loss Params
    beta: float = 4.0
    lambda_recon: float = 1.0 # Weight for reconstruction loss
    lambda_bb: float = 0.0    # Weight for Bounding Box supervision
    
    recon_loss_type: str = "mse" # "mse" or "bce"
    
    # Training Params
    lr: float = 0.001
    scheduler_patience: int = 10
    dataset: str = "pendulum"
    
    # Warmup
    warmup_epochs: int = 0

class BetaVAE(BaseCausalModel):
    def __init__(self, config: BetaVAEConfig):
        super().__init__()
        self.config = config
        
        # 1. Shared Encoder
        self.encoder_base = SharedEncoder(config.img_channels, config.img_size, config.latent_dim)
        
        # 2. VAE Heads
        self.fc_mu = nn.Linear(config.latent_dim, config.z_dim)
        self.fc_logvar = nn.Linear(config.latent_dim, config.z_dim)
        
        # 3. Shared Decoder
        self.fc_decode = nn.Linear(config.z_dim, config.latent_dim)
        self.decoder_base = SharedDecoder(
            config.img_channels, 
            config.img_size, 
            config.latent_dim, 
            self.encoder_base.flat_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels=None, mask=None) -> Dict[str, torch.Tensor]:
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        h_recon = self.fc_decode(z)
        x_recon_logits = self.decoder_base(h_recon)
        
        return {
            "x_recon_logits": x_recon_logits,
            "z": z, # For evaluation
            "mu": mu,
            "logvar": logvar,
            "mask": mask
        }

    def compute_loss(self, x, fwd_dict) -> Dict[str, torch.Tensor]:
        cfg = self.config
        
        # --- FIX: Scale Reconstruction Loss to be "Per Image" ---
        logits = fwd_dict["x_recon_logits"]
        batch_size, C, H, W = x.shape
        num_pixels = C * H * W
        
        if cfg.recon_loss_type == "bce":
            # reduction='mean' divides by (B*Pixels). Multiply by Pixels to get Mean-over-Batch Sum-over-Pixels.
            recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean') * num_pixels
        else:
            x_recon = torch.sigmoid(logits)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean') * num_pixels
            
        # Apply reconstruction weight
        recon_loss = recon_loss * cfg.lambda_recon
            
        # 2. KL Divergence
        mu, logvar = fwd_dict["mu"], fwd_dict["logvar"]
        # Sum over latent dim, Mean over batch (Matches recon_loss scale now)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        total = recon_loss + cfg.beta * kl_loss

        # 3. Bounding Box Loss
        bbox_loss = torch.tensor(0.0, device=x.device)
        if cfg.lambda_bb > 0 and fwd_dict.get("mask") is not None:
            mask = fwd_dict["mask"]
            x_recon = torch.sigmoid(logits)
            bbox_loss = ((x - x_recon) * mask).pow(2).sum() / batch_size
            bbox_loss = bbox_loss * cfg.lambda_bb
            
        total += bbox_loss
        
        return {"total": total, "recon": recon_loss, "kl": kl_loss, "bbox": bbox_loss}