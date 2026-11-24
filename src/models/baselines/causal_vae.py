import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

from .base_model import BaseCausalModel
from ..backbones import SharedEncoder, SharedDecoder

# --- Configuration ---
@dataclass
class CausalVAEConfig:
    # Architecture Params (Same as LANCA)
    img_channels: int = 3
    img_size: int = 96
    latent_dim: int = 64  # CNN output dim
    
    # CausalVAE Specific Params
    z1_dim: int = 4       # Number of Causal Concepts
    z2_dim: int = 4       # Dimension of each Concept
    
    # Loss Weights
    alpha: float = 0.3    # KL Weight for Exogenous (epsilon)
    beta: float = 1.0     # KL Weight for Endogenous (z)
    gamma: float = 1.0    # DAG constraint weight
    lambda_v: float = 0.001 # Variance scaling
    
    lambda_recon: float = 1.0 # Reconstruction Loss Weight
    lambda_bb: float = 0.0    # Bounding Box Supervision Weight
    
    # Training
    lr: float = 0.001
    lr_dag: float = 0.001
    scheduler_patience: int = 10
    recon_loss_type: str = "mse" # 'mse' or 'bce'
    dataset: str = "pendulum"
    
    # CausalVAE Logic
    unsupervised: bool = False # If True, skips Mask/Attention logic
    
    # Warmup
    warmup_epochs: int = 0
    
    # Model Internals
    inference: bool = False
    initial_dag: bool = True

# --- Helper Modules ---

class MaskLayer(nn.Module):
    def __init__(self, num_concepts, dim_per_concept):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim_per_concept = dim_per_concept
        self.nets = nn.ModuleList()
        
        for _ in range(num_concepts):
            net = nn.Sequential(
                nn.Linear(dim_per_concept, 32),
                nn.ELU(),
                nn.Linear(32, dim_per_concept),
            )
            self.nets.append(net)

    def mix(self, z):
        # z: [Batch, num_concepts, dim_per_concept]
        if self.dim_per_concept == 1 and z.dim() == 2:
            z = z.unsqueeze(-1)
            
        outputs = []
        for i in range(self.num_concepts):
            outputs.append(self.nets[i](z[:, i, :]))
        return torch.stack(outputs, dim=1)

class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.M = nn.Parameter(torch.randn(in_features, in_features) * 0.01)
        self.sigmd = torch.nn.Sigmoid()
        
    def forward(self, z, e):
        z_m = torch.matmul(z, self.M) 
        a = torch.bmm(z_m, e.permute(0, 2, 1)) 
        a = self.sigmd(a)
        A_softmax = F.softmax(a, dim=1)
        e_new = torch.bmm(A_softmax, e) 
        return e_new

class DagLayer(nn.Module):
    def __init__(self, num_concepts):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_concepts, num_concepts))
        nn.init.uniform_(self.A, -0.1, 0.1)
        self.I = nn.Parameter(torch.eye(num_concepts), requires_grad=False)

    def mask_z(self, x):
        x_perm = x.permute(0, 2, 1) 
        out = torch.matmul(x_perm, self.A.t())
        return out.permute(0, 2, 1)

    def mask_u(self, u):
        return torch.matmul(u, self.A.t())

    def calculate_dag(self, eps):
        W_inv = torch.linalg.pinv(self.I - self.A.t())
        eps_perm = eps.permute(0, 2, 1)
        z_perm = torch.matmul(eps_perm, W_inv.t())
        return z_perm.permute(0, 2, 1)

# --- Main Model ---

class CausalVAE(BaseCausalModel):
    def __init__(self, config: CausalVAEConfig):
        super().__init__()
        self.config = config
        
        # 1. Shared Encoder
        self.encoder_base = SharedEncoder(config.img_channels, config.img_size, config.latent_dim)
        
        # Bridge to Causal Latents
        self.z_dim = config.z1_dim * config.z2_dim
        self.fc_enc_mu = nn.Linear(config.latent_dim, self.z_dim)
        self.fc_enc_var = nn.Linear(config.latent_dim, self.z_dim)
        
        # 2. Causal Structure Layers
        self.dag_layer = DagLayer(config.z1_dim)
        self.mask_z = MaskLayer(config.z1_dim, config.z2_dim)
        self.attn = Attention(config.z2_dim)
        
        self.mask_u = nn.Linear(config.z1_dim, config.z1_dim, bias=False) 
        
        # 3. Shared Decoder
        self.fc_decode = nn.Linear(self.z_dim, config.latent_dim)
        self.decoder_base = SharedDecoder(
            config.img_channels, 
            config.img_size, 
            config.latent_dim, 
            self.encoder_base.flat_dim
        )

    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_enc_mu(h)
        logvar = self.fc_enc_var(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels=None, mask=None) -> Dict[str, torch.Tensor]:
        bs = x.size(0)
        
        # 1. Encode to Exogenous Noise (Epsilon)
        eps_mu, eps_logvar = self.encode(x)
        epsilon = self.reparameterize(eps_mu, eps_logvar)
        eps_reshaped = epsilon.view(bs, self.config.z1_dim, self.config.z2_dim)
        
        # 2. Calculate Endogenous Latents (Z)
        z_concepts = self.dag_layer.calculate_dag(eps_reshaped)
        
        # 3. Masking & Attention
        f_z = z_concepts
        g_u = None
        
        if not self.config.unsupervised and labels is not None:
            m_zm = self.dag_layer.mask_z(z_concepts)
            f_z_base = self.mask_z.mix(m_zm)
            e_tilde = self.attn(z_concepts, eps_reshaped)
            f_z = f_z_base + e_tilde
            m_u = self.dag_layer.mask_u(labels)
            g_u = self.mask_u(m_u)

        # 4. Decode
        z_flat = f_z.reshape(bs, -1) 
        h_recon = self.fc_decode(z_flat)
        x_recon_logits = self.decoder_base(h_recon)
        
        return {
            "x_recon_logits": x_recon_logits,
            "z": z_concepts,        
            "epsilon": epsilon,     
            "eps_mu": eps_mu,
            "eps_logvar": eps_logvar,
            "f_z": f_z,             
            "g_u": g_u,             
            "A": self.dag_layer.A,
            "labels": labels,
            "mask": mask
        }

    def compute_loss(self, x, fwd_dict) -> Dict[str, torch.Tensor]:
        cfg = self.config
        
        # --- FIX: Scale Reconstruction Loss ---
        logits = fwd_dict["x_recon_logits"]
        batch_size, C, H, W = x.shape
        num_pixels = C * H * W
        
        if cfg.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean') * num_pixels
        else:
            x_recon = torch.sigmoid(logits)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean') * num_pixels
            
        recon_loss = recon_loss * cfg.lambda_recon
            
        # 2. KL Divergence on Epsilon
        mu, logvar = fwd_dict["eps_mu"], fwd_dict["eps_logvar"]
        kl_eps = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. KL Divergence on Z
        z = fwd_dict["z"]
        if not cfg.unsupervised and "labels" in fwd_dict and fwd_dict["labels"] is not None:
            prior_mu = fwd_dict["labels"].unsqueeze(-1).expand_as(z)
        else:
            prior_mu = torch.zeros_like(z)
            
        # Using MSE for conditional prior as in paper/original implementation
        kl_z = F.mse_loss(z, prior_mu, reduction='mean') * (cfg.z1_dim * cfg.z2_dim) 
        
        # 4. DAG Constraint
        A = fwd_dict["A"]
        h_A = torch.trace(torch.matrix_exp(A * A)) - cfg.z1_dim
        
        # 5. Mask Loss
        mask_loss = torch.tensor(0.0, device=x.device)
        if not cfg.unsupervised and fwd_dict["g_u"] is not None and fwd_dict.get("labels") is not None:
            lbl = fwd_dict["labels"]
            # Scale these up to match magnitude if necessary, or rely on alpha/beta tuning
            l_u_loss = F.mse_loss(fwd_dict["g_u"], lbl)
            l_m_loss = F.mse_loss(fwd_dict["f_z"], fwd_dict["z"])
            mask_loss = l_u_loss + l_m_loss

        # 6. Bounding Box Supervision
        bbox_loss = torch.tensor(0.0, device=x.device)
        if cfg.lambda_bb > 0 and fwd_dict.get("mask") is not None:
            mask = fwd_dict["mask"]
            x_recon = torch.sigmoid(logits)
            bbox_loss = ((x - x_recon) * mask).pow(2).sum() / batch_size
            bbox_loss = bbox_loss * cfg.lambda_bb

        total = recon_loss + (cfg.alpha * kl_eps) + (cfg.beta * kl_z) + \
                (cfg.gamma * h_A) + mask_loss + bbox_loss
        
        return {
            "total": total, 
            "recon": recon_loss, 
            "kl_eps": kl_eps, 
            "kl_z": kl_z,
            "dag": h_A, 
            "mask": mask_loss,
            "bbox": bbox_loss
        }

    def perform_intervention(self, x, interventions: Dict[int, float]):
        self.eval()
        with torch.no_grad():
            bs = x.size(0)
            mu, _ = self.encode(x)
            eps = mu.view(bs, self.config.z1_dim, self.config.z2_dim)
            z_base = self.dag_layer.calculate_dag(eps)
            z_new = z_base.clone()
            
            for _ in range(self.config.z1_dim):
                z_parents = self.dag_layer.mask_z(z_new)
                for i in range(self.config.z1_dim):
                    if i in interventions:
                        z_new[:, i, :] = interventions[i]
                    else:
                        z_new[:, i, :] = z_parents[:, i, :] + eps[:, i, :]
            
            z_flat = z_new.reshape(bs, -1)
            h_recon = self.fc_decode(z_flat)
            return torch.sigmoid(self.decoder_base(h_recon))