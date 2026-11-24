import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import wandb
import json
from dataclasses import asdict, is_dataclass
from tqdm import tqdm

class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        config, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        device: str,
        save_dir: str
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        self.target_beta = getattr(config, 'beta', 1.0)

        # --- 1. SEPARATE OPTIMIZER GROUPS ---
        # We need to distinguish between:
        # 1. Main Network (Encoder/Decoder/SCM functions) -> config.lr
        # 2. DAG Edges (The U matrix probabilities)       -> config.lr_dag
        # 3. DAG Permutation (The P matrix sorting)       -> config.lr_dag_perm (NEW)

        lr_main = config.lr
        lr_dag_edges = getattr(config, 'lr_dag', config.lr)
        # Default to lr_dag if lr_dag_perm is not set
        lr_dag_perm = getattr(config, 'lr_dag_perm', lr_dag_edges) 

        main_params = []
        dag_edge_params = []
        dag_perm_params = []

        for name, param in self.model.named_parameters():
            if "discriminator" in name: continue 
            
            if "dag_layer" in name:
                if "perm_weights" in name:
                    dag_perm_params.append(param)
                else:
                    # Includes edge_log_params
                    dag_edge_params.append(param)
            else:
                main_params.append(param)
        
        param_groups = [{'params': main_params, 'lr': lr_main}]
        
        if dag_edge_params:
            param_groups.append({'params': dag_edge_params, 'lr': lr_dag_edges})
            
        if dag_perm_params:
            print(f"Optimizer: Split Permutation Params (LR={lr_dag_perm})")
            param_groups.append({'params': dag_perm_params, 'lr': lr_dag_perm})
            
        self.optimizer = optim.Adam(param_groups, lr=lr_main)
        
        # Discriminator Optimizer (FactorVAE)
        self.optimizer_dis = None
        if hasattr(self.model, "discriminator"):
            self.optimizer_dis = optim.Adam(
                self.model.discriminator.parameters(), 
                lr=getattr(config, 'lr_dis', 1e-4)
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=config.scheduler_patience
        )
        
        try:
            batch = next(iter(val_loader))
            if isinstance(batch, list):
                self.fixed_val_batch = batch[0].to(device)
            else:
                self.fixed_val_batch = batch[0].to(device)
        except StopIteration:
            self.fixed_val_batch = None
            
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # Dual Ascent Tracking
        self.independence_tol = 0.05
        self.rho_prior = 2.0
        self.max_lambda_prior = 100.0
        
        # Save Config
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        self.save_config()

    def save_config(self):
        config_path = os.path.join(self.save_dir, "config.json")
        if is_dataclass(self.config):
            new_config_dict = asdict(self.config)
        elif hasattr(self.config, '__dict__'):
            new_config_dict = self.config.__dict__
        else:
            new_config_dict = self.config
            
        final_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f: existing = json.load(f)
                final_config.update(existing)
            except Exception: pass
        
        final_config.update(new_config_dict)
        with open(config_path, "w") as f: json.dump(final_config, f, indent=4)

    def update_warmup_params(self, epoch):
        """
        Anneals hyperparameters to prevent early lock-in.
        1. Beta (VAE): 0 -> Target
        2. Lambda DAG (Sparsity): 0 -> Target
        3. Lambda Sort (Permutation): 0 -> Target
        4. Tau Edges (Temperature): Start Tau -> End Tau (Sharpening)
        """
        if not hasattr(self.config, 'warmup_epochs') or self.config.warmup_epochs <= 0:
            return

        # Linear progress (0.0 to 1.0)
        progress = min(1.0, epoch / self.config.warmup_epochs)
        
        # --- TUNABLE SPARSITY DELAY ---
        # Default to 0.5 if not specified in config
        sparsity_start_phase = getattr(self.config, 'sparsity_delay', 0.5)
        
        if progress < sparsity_start_phase:
            # Phase 1: Pure Structure Learning
            current_dag_lambda = 0.0
        else:
            # Phase 2: Anneal Sparsity (0 -> Target)
            # Re-normalize progress to be 0.0 to 1.0 within the remaining window
            denom = 1.0 - sparsity_start_phase
            if denom == 0: denom = 1e-6 # Avoid div/0
            
            sparsity_progress = (progress - sparsity_start_phase) / denom
            target_dag = getattr(self.config, 'lambda_dag', 0.05)
            current_dag_lambda = target_dag * sparsity_progress
            
        if hasattr(self.model.config, 'lambda_dag'):
            self.model.config.lambda_dag = current_dag_lambda

        # 1. Beta Annealing
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'beta'):
            self.model.config.beta = self.target_beta * progress

        # 2. DAG Sparsity Annealing
        if hasattr(self.model.config, 'lambda_dag'):
            target_dag = getattr(self.config, 'lambda_dag', 0.05)
            self.model.config.lambda_dag = target_dag * progress

        # 3. Sort Regularization Annealing
        if hasattr(self.model.config, 'lambda_sort'):
            target_sort = getattr(self.config, 'lambda_sort', 1.0)
            self.model.config.lambda_sort = target_sort * progress

        # --- 2. SPARSITY DELAY (Prevent Empty Graph) ---
        target_dag = getattr(self.config, 'lambda_dag', 0.05)
        delay_frac = getattr(self.config, 'sparsity_delay', 0.5)
        
        if progress < delay_frac:
            # Phase 1: Structure Discovery (No penalty)
            current_dag = 0.0
        else:
            # Phase 2: Pruning (Anneal 0 -> Target)
            # Normalize progress to [0, 1] range within the remaining window
            adj_progress = (progress - delay_frac) / (1.0 - delay_frac)
            current_dag = target_dag * adj_progress
            
        if hasattr(self.model.config, 'lambda_dag'):
            self.model.config.lambda_dag = current_dag

        # --- 3. TEMPERATURE ANNEALING (Soft -> Hard) ---
        if hasattr(self.model.config, 'tau_edges'):
            start_tau = getattr(self.config, 'tau_edges', 5.0) # Config value is START
            end_tau = 0.5 # Hard floor
            
            # Exponential decay is often better than linear for temperature
            # current_tau = start_tau * ( (end_tau/start_tau)**progress )
            
            # Or Simple Linear (easier to tune):
            current_tau = start_tau - (start_tau - end_tau) * progress
            self.model.config.tau_edges = max(end_tau, current_tau)
        # # 4. Tau Annealing (Sharpening)
        # # Use the config value as the starting point
        # if hasattr(self.model.config, 'tau_edges'):
        #     start_tau = getattr(self.config, 'tau_edges', 1.0)
        #     end_tau = 0.5 # Hard lower limit to ensure discreteness
            
        #     # Don't anneal if start is already lower than end (sanity check)
        #     if start_tau > end_tau:
        #         current_tau = start_tau - (start_tau - end_tau) * progress
        #     else:
        #         current_tau = start_tau
                
        #     self.model.config.tau_edges = current_tau

        if wandb.run is not None:
            wandb.log({
                "train/beta": self.model.config.beta,
                "train/lambda_dag": getattr(self.model.config, 'lambda_dag', 0),
                "train/lambda_sort": getattr(self.model.config, 'lambda_sort', 0),
                "train/tau_edges": getattr(self.model.config, 'tau_edges', 1.0)
            })

    def update_dual_ascent(self, epoch, val_metrics):
        current_val = val_metrics.get('prior', val_metrics.get('hsic', 0.0))
        violation = max(0, current_val - self.independence_tol)
        
        if violation > 0 and epoch > self.config.warmup_epochs:
            old_lambda = getattr(self.model.config, 'lambda_prior', 1.0)
            new_lambda = min(self.max_lambda_prior, old_lambda * (1.0 + self.rho_prior * violation))
            self.model.config.lambda_prior = new_lambda
            
            if wandb.run is not None:
                wandb.log({"train/lambda_prior_dynamic": new_lambda})

    def pretrain_dag_supervised(self):
        # CausalVAE Logic (Skipped if not applicable)
        is_causal_vae = (getattr(self.config, 'model_name', '') == 'causal_vae')
        is_supervised = hasattr(self.config, 'unsupervised') and (not self.config.unsupervised)
        
        if not (is_causal_vae and is_supervised):
            return

        print("\n=== Starting Structural Pretraining (Supervised CausalVAE) ===")
        dag_params = [p for n, p in self.model.named_parameters() if "dag_layer" in n]
        if not dag_params: return
        
        dag_optimizer = optim.Adam(dag_params, lr=getattr(self.config, 'lr_dag', 1e-3))
        
        rho = 1.0     
        alpha = 0.0   
        h_prev = float('inf')
        epochs = 10 
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_h = 0
            for batch in self.train_loader:
                if isinstance(batch, list) and len(batch) >= 2: _, labels = batch[0], batch[1]
                else: continue 
                
                labels = labels.to(self.device)
                dag_optimizer.zero_grad()
                
                # Specific CausalVAE access pattern
                A = self.model.dag_layer.A
                u_recon = self.model.dag_layer.mask_u(labels)
                mse_u = torch.nn.functional.mse_loss(u_recon, labels, reduction='mean')
                
                d = A.shape[0]
                h_A = torch.trace(torch.matrix_exp(A * A)) - d
                loss = mse_u + alpha * h_A + 0.5 * rho * h_A * h_A
                
                loss.backward()
                dag_optimizer.step()
                total_loss += loss.item()
                total_h += h_A.item()
            
            avg_h = total_h / len(self.train_loader)
            if avg_h > 0.25 * h_prev: rho *= 10.0
            alpha += rho * avg_h
            h_prev = avg_h
            if rho > 1e20: rho = 1e20

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        self.update_warmup_params(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            if isinstance(batch, list):
                if len(batch) == 3: x, labels, mask = batch
                elif len(batch) == 2: x, labels = batch; mask = None
            else: x = batch[0]; labels = None; mask = None

            x = x.to(self.device)
            if labels is not None: labels = labels.to(self.device)
            if mask is not None: mask = mask.to(self.device)

            self.optimizer.zero_grad()
            
            # Handle mask argument if model supports it
            kwargs = {}
            if "mask" in self.model.forward.__code__.co_varnames and mask is not None:
                kwargs['mask'] = mask
            
            fwd_dict = self.model(x, labels=labels, **kwargs)
            losses = self.model.compute_loss(x, fwd_dict)
            
            if not torch.isfinite(losses["total"]):
                continue
                
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # FactorVAE Discriminator Step
            if self.optimizer_dis is not None and hasattr(self.model, "compute_discriminator_loss"):
                self.optimizer_dis.zero_grad()
                dis_loss = self.model.compute_discriminator_loss(fwd_dict["z"])
                dis_loss.backward()
                self.optimizer_dis.step()

            total_loss += losses["total"].item()
            pbar.set_postfix({"loss": losses["total"].item()})
            
            if wandb.run is not None:
                wandb.log({f"train/batch/{k}": v.item() for k, v in losses.items()})

        return total_loss / len(self.train_loader)

    def val_epoch(self, epoch):
        self.model.eval()
        loss_accumulator = {}
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, list):
                    if len(batch) == 3: x, labels, mask = batch
                    elif len(batch) == 2: x, labels = batch; mask = None
                else: x = batch[0]; labels = None; mask = None
                    
                x = x.to(self.device)
                if labels is not None: labels = labels.to(self.device)
                if mask is not None: mask = mask.to(self.device)
                
                kwargs = {}
                if "mask" in self.model.forward.__code__.co_varnames and mask is not None:
                    kwargs['mask'] = mask

                fwd_dict = self.model(x, labels=labels, **kwargs)
                losses = self.model.compute_loss(x, fwd_dict)
                for k, v in losses.items():
                    loss_accumulator[k] = loss_accumulator.get(k, 0.0) + v.item()
        
        avg_losses = {k: v / len(self.val_loader) for k, v in loss_accumulator.items()}
        
        if wandb.run is not None:
            wandb.log({f"val/epoch/{k}": v for k, v in avg_losses.items()})
        
        return avg_losses

    def save_visuals(self, epoch):
        if self.fixed_val_batch is None: return
        self.model.eval()
        with torch.no_grad():
            # Check signature
            args = self.model.forward.__code__.co_varnames
            kwargs = {}
            if "labels" in args: 
                dim = getattr(self.model.config, 'num_causal_vars', 4)
                kwargs['labels'] = torch.zeros(self.fixed_val_batch.size(0), dim).to(self.device)
            
            fwd_dict = self.model(self.fixed_val_batch, **kwargs)
            # Handle models that might output raw logits vs sigmoid
            if "x_recon_logits" in fwd_dict:
                recon = torch.sigmoid(fwd_dict["x_recon_logits"])
            else:
                recon = self.fixed_val_batch # Fallback if key missing
            
            n = min(8, self.fixed_val_batch.size(0))
            comp = torch.cat([self.fixed_val_batch[:n], recon[:n]])
            grid = vutils.make_grid(comp, nrow=n)
            vutils.save_image(grid, os.path.join(self.save_dir, "images", f"recon_{epoch}.png"))


    def fit(self, num_epochs, patience, save_img_freq):
        self.pretrain_dag_supervised()
        
        # Reset best_loss to infinity to ensure we don't keep a "cheap" warmup model
        self.best_val_loss = float('inf') 
        
        warmup_epochs = getattr(self.config, 'warmup_epochs', 0)

        for epoch in range(1, num_epochs + 1):
            _ = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            
            self.update_dual_ascent(epoch, val_metrics)
            
            # --- Scheduler Step ---
            # Ideally, don't decay LR during warmup based on "fake" low loss
            if epoch > warmup_epochs:
                self.scheduler.step(val_metrics["total"])
            
            if epoch % save_img_freq == 0:
                self.save_visuals(epoch)
            
            # Always save last model
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model.pth"))
            
            # --- CRITICAL FIX: LOGIC SPLIT ---
            if epoch <= warmup_epochs:
                # DURING WARMUP:
                # 1. Do not count towards Early Stopping
                # 2. Do not save as "best_model" (it's "cheating" with 0 penalty)
                self.epochs_no_improve = 0
                
                # Optional: If you want to see progress, you can save a separate checkpoint
                # torch.save(self.model.state_dict(), os.path.join(self.save_dir, "warmup_model.pth"))
                
            else:
                # AFTER WARMUP:
                # Standard Early Stopping & Best Model Logic
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
                    
                    with open(os.path.join(self.save_dir, "metrics.json"), 'w') as f:
                        json.dump(val_metrics, f, indent=4)
                else:
                    self.epochs_no_improve += 1
                    
                if self.epochs_no_improve >= patience:
                    print(f"Early Stopping at Epoch {epoch} (Patience {patience} exceeded after Warmup)")
                    break