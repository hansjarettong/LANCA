import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .backbones import SharedEncoder, SharedDecoder
from .components.dag_layer import DifferentiableDAGLayer
from .components.scm import SCMLayer
from .components.hsic import hsic_kernel
from src.training.losses import compute_mmd

@dataclass
class LANCAConfig:
    img_channels: int = 3
    img_size: int = 96
    latent_dim: int = 64
    num_causal_vars: int = 4

    dataset: str = "pendulum"
    lr: float = 0.001
    lr_dag: float = 0.001
    scheduler_patience: int = 10
    warmup_epochs: int = 20
    
    sparsity_delay: float = 0.5    # Fraction of warmup epochs to keep lambda_dag = 0
    lr_dag_perm: float = 0.005     # Separate learning rate for permutation learning
    
    scm_hidden_dim: int = 64
    scm_layers: int = 2
    scm_activation: str = "tanh"
    
    recon_loss_type: str = "mse"
    prior_mode: str = "mmd"

    mmd_kernel: str = "imq" # "rbf" or "imq"
    lambda_prior: float = 1.0
    
    lambda_recon: float = 1.0
    lambda_recon_scm: float = 1.0
    lambda_dag: float = 0.05
    lambda_sort: float = 1.0
    tau_edges: float = 5.0
    
    lambda_l1_scm: float = 0.0
    lambda_bb: float = 0.0
    
    dag_sparsity_prior: float = 0.01
    hsic_sigma: float = 1.0
    
    poly_degree: int = 2
    trig_terms: bool = True
    interaction_max_degree: int = 2
    lambda_hsic: float = 1.0   
    lambda_hsic_mutual: float = 1.0

    # R2 Revisions (R2.2.1, R2.2.2)
    stochastic_encoder: bool = False
    beta_kl: float = 0.0
    prior_dist: str = "gaussian" # "gaussian", "uniform", "laplace"

class CausalSCMRegularizer(nn.Module):
    def __init__(self, config: LANCAConfig):
        super().__init__()
        self.config = config
        self.dag_layer = DifferentiableDAGLayer(config.num_causal_vars)
        self.anm_nets = nn.ModuleList([
            SCMLayer(config) for _ in range(config.num_causal_vars)
        ])

    def compute_residuals(self, z, A):
        residuals = []
        parents_list = []
        B = z.shape[0]
        
        for i in range(self.config.num_causal_vars):
            parent_mask = A[:, i].unsqueeze(0)
            z_parents = z * parent_mask.expand(B, -1)
            z_hat_i = self.anm_nets[i](z_parents).squeeze(-1)
            eps_i = z[:, i] - z_hat_i
            
            residuals.append(eps_i.unsqueeze(-1))
            parents_list.append(z_parents)
            
        return residuals, parents_list

    def propagate_intervention(self, z_observed, A, causal_order, interventions):
        B = z_observed.shape[0]
        epsilon = torch.zeros_like(z_observed)
        z_new = z_observed.clone()
        
        # 1. Abduction
        for i in range(self.config.num_causal_vars):
            parent_mask = A[:, i].unsqueeze(0)
            z_parents = z_observed * parent_mask.expand(B, -1)
            z_hat_i = self.anm_nets[i](z_parents).squeeze(-1)
            epsilon[:, i] = z_observed[:, i] - z_hat_i

        # 2. Action & 3. Prediction
        for i in causal_order:
            if i in interventions:
                z_new = z_new.clone() 
                z_new[:, i] = interventions[i]
            else:
                parent_mask = A[:, i].unsqueeze(0)
                z_parents_new = z_new * parent_mask.expand(B, -1)
                z_hat_new = self.anm_nets[i](z_parents_new).squeeze(-1)
                z_new = z_new.clone()
                z_new[:, i] = z_hat_new + epsilon[:, i]
                
        return z_new

class LANCA(nn.Module):
    def __init__(self, config: LANCAConfig):
        super().__init__()
        self.config = config
        self.encoder_base = SharedEncoder(config.img_channels, config.img_size, config.latent_dim)
        if getattr(config, 'stochastic_encoder', False):
            self.fc_mu = nn.Linear(config.latent_dim, config.num_causal_vars)
            self.fc_logvar = nn.Linear(config.latent_dim, config.num_causal_vars)
        else:
            self.fc_z = nn.Linear(config.latent_dim, config.num_causal_vars)
        self.scm = CausalSCMRegularizer(config)
        self.fc_decode_in = nn.Linear(config.num_causal_vars, config.latent_dim)
        self.decoder_base = SharedDecoder(
            config.img_channels, 
            config.img_size, 
            config.latent_dim, 
            self.encoder_base.flat_dim
        )

    def get_functional_graph(self, threshold=1e-3):
        """
        Returns the adjacency matrix refined by functional pruning.
        It sets A[j, i] = 0 if the SCM for node i has near-zero weights for input j.
        This addresses the ambiguity where the DAG layer predicts an edge but the
        function effectively ignores it.
        """
        # 1. Get Structural Graph (Binary)
        A_struct, order = self.scm.dag_layer.get_dag_and_order(hard=True)
        A_refined = A_struct.clone().detach()
        
        # 2. Prune based on SCM weights
        # SCMLayer.net[0] is the input Linear layer: (hidden_dim, num_vars)
        for i in range(self.config.num_causal_vars):
            # Weights for mechanism of node i
            # Shape: (hidden, input_dim) -> input_dim is num_causal_vars
            w = self.scm.anm_nets[i].net[0].weight.detach().abs()
            
            # Sum weights across hidden neurons for each input variable
            # Shape: (num_causal_vars,)
            w_sum = w.sum(dim=0)
            
            # If total weight for input j is small, prune edge j -> i
            for j in range(self.config.num_causal_vars):
                if A_refined[j, i] > 0 and w_sum[j] < threshold:
                    A_refined[j, i] = 0.0
                    
        return A_refined, order

    def perform_intervention(self, x, interventions: Dict[int, float]):
        self.eval()
        with torch.no_grad():
            h = self.encoder_base(x)
            if getattr(self.config, 'stochastic_encoder', False):
                z_observed = self.fc_mu(h)
            else:
                z_observed = self.fc_z(h)
            A_hard, causal_order_idx = self.scm.dag_layer.get_dag_and_order(hard=True)
            if torch.is_tensor(causal_order_idx): causal_order = causal_order_idx.tolist()
            else: causal_order = causal_order_idx

            z_counterfactual = self.scm.propagate_intervention(
                z_observed, A_hard, causal_order, interventions
            )
            h_recon = self.fc_decode_in(z_counterfactual)
            x_counterfactual_logits = self.decoder_base(h_recon)
            return torch.sigmoid(x_counterfactual_logits)

    def forward(self, x, labels=None, mask=None) -> Dict[str, torch.Tensor]:
        h = self.encoder_base(x)
        mu, logvar = None, None
        if getattr(self.config, 'stochastic_encoder', False):
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
        else:
            z = self.fc_z(h)
        h_recon = self.fc_decode_in(z)
        x_recon_logits = self.decoder_base(h_recon)
        
        current_tau = getattr(self.config, 'tau_edges', 1.0)
        
        A_soft, _ = self.scm.dag_layer.get_dag_and_order(hard=False, tau_edges=current_tau) 
        residuals, parents = self.scm.compute_residuals(z, A_soft)
        
        A_hard, causal_order_idx = self.scm.dag_layer.get_dag_and_order(hard=True, tau_edges=current_tau)
        if torch.is_tensor(causal_order_idx): causal_order = causal_order_idx.tolist()
        else: causal_order = causal_order_idx
            
        z_scm = self.scm.propagate_intervention(z, A_hard, causal_order, interventions={})
        h_recon_scm = self.fc_decode_in(z_scm)
        x_recon_scm_logits = self.decoder_base(h_recon_scm)
        
        return {
            "x_recon_logits": x_recon_logits,
            "x_recon_scm_logits": x_recon_scm_logits,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "residuals": residuals,
            "parents": parents,
            "A_soft": A_soft,
            "labels": labels,
            "mask": mask
        }

    def compute_loss(self, x, fwd_dict) -> Dict[str, torch.Tensor]:
        cfg = self.config
        losses = {}
        
        logits = fwd_dict["x_recon_logits"]
        batch_size, C, H, W = x.shape
        num_pixels = C * H * W
        
        if cfg.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean') * num_pixels
        else:
            x_recon = torch.sigmoid(logits)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean') * num_pixels
        losses["recon"] = recon_loss * cfg.lambda_recon
        
        logits_scm = fwd_dict["x_recon_scm_logits"]
        if cfg.recon_loss_type == "bce":
            recon_scm_loss = F.binary_cross_entropy_with_logits(logits_scm, x, reduction='mean') * num_pixels
        else:
            x_recon_scm = torch.sigmoid(logits_scm)
            recon_scm_loss = F.mse_loss(x_recon_scm, x, reduction='mean') * num_pixels
        losses["recon_scm"] = recon_scm_loss * cfg.lambda_recon_scm
        
        # Stochastic VAE KL Divergence Regularization (R2.2.2)
        if getattr(cfg, 'stochastic_encoder', False) and getattr(cfg, 'beta_kl', 0.0) > 0:
            mu = fwd_dict.get("mu")
            logvar = fwd_dict.get("logvar")
            if mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                losses["kl"] = kl_loss * cfg.beta_kl

        if cfg.prior_mode == "mmd":
            epsilon_est = torch.cat(fwd_dict["residuals"], dim=1)
            dist_name = getattr(cfg, 'prior_dist', 'gaussian')
            if dist_name == "uniform":
                # Uniform U[-sqrt(3), sqrt(3)] has mean 0, variance 1
                target_epsilon = (torch.rand_like(epsilon_est) * 2.0 - 1.0) * (3.0 ** 0.5)
            elif dist_name == "laplace":
                # Laplace with mean 0, variance 1: scale = 1/sqrt(2)
                scale = 1.0 / (2.0 ** 0.5)
                u_rand = torch.rand_like(epsilon_est).clamp(1e-6, 1.0 - 1e-6)
                target_epsilon = -scale * torch.sign(u_rand - 0.5) * torch.log(1.0 - 2.0 * torch.abs(u_rand - 0.5))
            else:
                target_epsilon = torch.randn_like(epsilon_est)
            losses["prior"] = compute_mmd(epsilon_est, target_epsilon, kernel_type=cfg.mmd_kernel) * cfg.lambda_prior

        elif cfg.prior_mode == "hsic":
            hsic_vals = []
            for i in range(cfg.num_causal_vars):
                res = fwd_dict["residuals"][i]
                par = fwd_dict["parents"][i]
                if par.std() > 1e-6: 
                    hsic_vals.append(hsic_kernel(par, res, sigma=cfg.hsic_sigma))
            val = torch.mean(torch.stack(hsic_vals)) if hsic_vals else torch.tensor(0.0, device=x.device)
            losses["prior"] = val * cfg.lambda_prior

        current_tau = getattr(cfg, 'tau_edges', 1.0)
        sparsity = self.scm.dag_layer.sparsity_regularizer(cfg.dag_sparsity_prior, tau_edges=current_tau)
        losses["dag"] = sparsity * cfg.lambda_dag
        
        sort_reg = self.scm.dag_layer.permutation_regularizer()
        losses["sort"] = sort_reg * getattr(cfg, 'lambda_sort', 0.0)
        
        if cfg.lambda_l1_scm > 0:
            l1_loss = torch.tensor(0.0, device=x.device)
            for net in self.scm.anm_nets:
                for name, param in net.named_parameters():
                    if 'weight' in name:
                        l1_loss += torch.norm(param, p=1)
            losses["l1_scm"] = l1_loss * cfg.lambda_l1_scm

        losses["bbox"] = torch.tensor(0.0, device=x.device)
        if cfg.lambda_bb > 0 and fwd_dict.get("mask") is not None:
            mask = fwd_dict["mask"]
            x_recon = torch.sigmoid(logits)
            bbox_loss = ((x_recon - x) * mask).pow(2).sum() / x.size(0)
            losses["bbox"] = bbox_loss * cfg.lambda_bb

        losses["total"] = sum(losses.values())

        return losses