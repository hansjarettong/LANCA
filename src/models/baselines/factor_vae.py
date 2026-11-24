import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

from .base_model import BaseCausalModel
from ..backbones import SharedEncoder, SharedDecoder

@dataclass
class FactorVAEConfig:
    img_channels: int = 3
    img_size: int = 96
    latent_dim: int = 64  # CNN bottleneck
    z_dim: int = 10       # VAE latent size
    
    gamma: float = 10.0   # Weight for Total Correlation (TC) loss
    lambda_bb: float = 0.0 # Weight for Bounding Box supervision

    lr: float = 0.001     # VAE Learning Rate
    lr_dis: float = 1e-4  # Discriminator Learning Rate (Original default)
    scheduler_patience: int = 10

    dataset: str = "pendulum"
    
    recon_loss_type: str = "mse"

class Discriminator(nn.Module):
    """
    The exact Discriminator architecture from CAN-VAE/factor_vae.py
    """
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 2) # Logits: [Real (0), Fake (1)]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class FactorVAE(BaseCausalModel):
    def __init__(self, config: FactorVAEConfig):
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
        
        # 4. Discriminator
        self.discriminator = Discriminator(config.z_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def permute_dims(self, z):
        """
        Shuffles latent variables across the batch dimension (from original train_factor_vae.py)
        to approximate the product of marginals.
        """
        B, D = z.size()
        z_perm = torch.zeros_like(z)
        for d in range(D):
            idx = torch.randperm(B)
            z_perm[:, d] = z[idx, d]
        return z_perm

    def forward(self, x, labels=None, mask=None) -> Dict[str, torch.Tensor]:
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        h_recon = self.fc_decode(z)
        x_recon_logits = self.decoder_base(h_recon)
        
        return {
            "x_recon_logits": x_recon_logits,
            "z": z, # z_sampled
            "mu": mu,
            "logvar": logvar,
            "mask": mask
        }

    def compute_loss(self, x, fwd_dict) -> Dict[str, torch.Tensor]:
        """
        Computes VAE loss: Recon + KL + gamma * TC
        """
        cfg = self.config
        
        # --- FIX: Scale Reconstruction Loss to be "Per Image" ---
        logits = fwd_dict["x_recon_logits"]
        batch_size, C, H, W = x.shape
        num_pixels = C * H * W
        
        if cfg.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean') * num_pixels
        else:
            x_recon = torch.sigmoid(logits)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean') * num_pixels
            
        # 2. KL Divergence (Gaussian Prior)
        mu, logvar = fwd_dict["mu"], fwd_dict["logvar"]
        # Sum over latent, mean over batch (matches scale of recon now)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # 3. TC Estimate (using density ratio trick)
        z = fwd_dict["z"]
        d_logits = self.discriminator(z)
        
        # Clamp to prevent NaN/Inf
        d_logits = torch.clamp(d_logits, min=-20., max=20.)
        
        # TC = E[log(D(z)) - log(1-D(z))] = E[logit_real - logit_fake]
        tc_loss = (d_logits[:, 0] - d_logits[:, 1]).mean()
        
        total = recon_loss + kl_loss + cfg.gamma * tc_loss

        # 4. Bounding Box Loss
        bbox_loss = torch.tensor(0.0, device=x.device)
        if cfg.lambda_bb > 0 and fwd_dict.get("mask") is not None:
            mask = fwd_dict["mask"]
            x_recon = torch.sigmoid(logits)
            bbox_loss = ((x - x_recon) * mask).pow(2).sum() / batch_size
            bbox_loss = bbox_loss * cfg.lambda_bb
            
        total += bbox_loss
        
        return {"total": total, "recon": recon_loss, "kl": kl_loss, "tc": tc_loss, "bbox": bbox_loss}

    def compute_discriminator_loss(self, z):
        """
        Separate loss for the Discriminator optimizer.
        """
        z = z.detach() # Stop gradients to VAE
        z_perm = self.permute_dims(z).detach()
        
        d_logits_real = self.discriminator(z)
        d_logits_fake = self.discriminator(z_perm)
        
        # Targets: Real=0, Fake=1 (matching original script)
        labels_real = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        labels_fake = torch.ones(z.size(0), dtype=torch.long, device=z.device)
        
        loss_real = F.cross_entropy(d_logits_real, labels_real)
        loss_fake = F.cross_entropy(d_logits_fake, labels_fake)
        
        return 0.5 * (loss_real + loss_fake)