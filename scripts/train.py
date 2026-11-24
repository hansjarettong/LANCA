import sys
import os
import argparse
import yaml
import torch
import wandb
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import Trainer
from src.models.lanca import LANCA, LANCAConfig
from src.models.baselines.beta_vae import BetaVAE, BetaVAEConfig
from src.models.baselines.causal_vae import CausalVAE, CausalVAEConfig
from src.models.baselines.factor_vae import FactorVAE, FactorVAEConfig
from src.utils.data_loader import CausalImageDataset

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def get_model(model_name, args_dict):
    def filter_args(cfg_class, args):
        valid_keys = cfg_class.__annotations__.keys()
        return {k: v for k, v in args.items() if k in valid_keys}

    if model_name == 'lanca':
        filtered = filter_args(LANCAConfig, args_dict)
        cfg = LANCAConfig(**filtered)
        return LANCA(cfg), cfg
        
    elif model_name == 'beta_vae':
        filtered = filter_args(BetaVAEConfig, args_dict)
        cfg = BetaVAEConfig(**filtered)
        return BetaVAE(cfg), cfg
        
    elif model_name == 'factor_vae':
        filtered = filter_args(FactorVAEConfig, args_dict)
        cfg = FactorVAEConfig(**filtered)
        return FactorVAE(cfg), cfg
        
    elif model_name == 'causal_vae':
        filtered = filter_args(CausalVAEConfig, args_dict)
        cfg = CausalVAEConfig(**filtered)
        return CausalVAE(cfg), cfg
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    
    # General Args
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--dataset', type=str, default='pendulum')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--config', type=str, help='Path to YAML config override')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_dag', type=float, default=0.001)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    
    # Architecture Args
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--recon_loss_type', type=str, default='bce')
    
    # Model Specifics (LANCA)
    parser.add_argument('--num_causal_vars', type=int, default=4)
    parser.add_argument('--poly_degree', type=int, default=2)
    parser.add_argument('--trig_terms', type=str2bool, default=True)
    parser.add_argument('--interaction_max_degree', type=int, default=2)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_recon_scm', type=float, default=1.0)
    
    # SCM Architecture
    parser.add_argument('--scm_hidden_dim', type=int, default=64)
    parser.add_argument('--scm_layers', type=int, default=2)
    parser.add_argument('--scm_activation', type=str, default='tanh')

    # Prior Settings
    parser.add_argument('--prior_mode', type=str, default='mmd')
    parser.add_argument('--mmd_kernel', type=str, default='imq', choices=['rbf', 'imq'], help="Kernel type for MMD loss")
    parser.add_argument('--lambda_prior', type=float, default=1.0)
    parser.add_argument('--prior_dist', type=str, default='gaussian', choices=['gaussian', 'uniform', 'laplace'], help="Residual prior distribution (R2.2.1)")
    
    # Stochastic Encoder Ablation (R2.2.2)
    parser.add_argument('--stochastic_encoder', type=str2bool, default=False, help="Use stochastic VAE encoder instead of deterministic WAE")
    parser.add_argument('--beta_kl', type=float, default=0.0, help="KL divergence regularization weight for stochastic encoder")
    
    # Structure Learning
    parser.add_argument('--lambda_hsic', type=float, default=1.0)
    parser.add_argument('--lambda_hsic_mutual', type=float, default=0.0)
    parser.add_argument('--lambda_dag', type=float, default=0.05)
    
    # --- NEW: Topological Params ---
    parser.add_argument('--lambda_sort', type=float, default=1.0, help="Weight for permutation matrix entropy regularization")
    parser.add_argument('--tau_edges', type=float, default=1.0, help="Initial temperature for edge sigmoid")
    
    parser.add_argument('--lambda_l1_scm', type=float, default=0.0)
    parser.add_argument('--dag_sparsity_prior', type=float, default=0.01)
    parser.add_argument('--hsic_sigma', type=float, default=1.0)
    parser.add_argument('--lambda_bb', type=float, default=0.0)
    
    # Baselines
    parser.add_argument('--z_dim', type=int, default=4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--lr_dis', type=float, default=1e-4)
    parser.add_argument('--z1_dim', type=int, default=4)
    parser.add_argument('--z2_dim', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--unsupervised', type=str2bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--lambda_v', type=float, default=0.001)
    parser.add_argument('--inference', type=str2bool, default=False)
    parser.add_argument('--initial_dag', type=str2bool, default=True)

    parser.add_argument('--lr_dag_perm', type=float, default=0.005)
    parser.add_argument('--sparsity_delay', type=float, default=0.5)

    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"[Train] Warning: Ignoring unrecognized arguments: {unknown}")

    # --- Load Config & Overwrite Args ---
    config_dict = vars(args).copy()
    if args.config:
        with open(args.config, 'r') as f:
            loaded_config = yaml.safe_load(f)
            config_dict.update(loaded_config)
            for k, v in loaded_config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
    
    # Robust Model Selection
    potential_model = args.model or config_dict.get('model') or config_dict.get('model_name')
    if not potential_model:
        parser.error("Argument --model is required")
        
    args.model = potential_model
    config_dict['model'] = potential_model

    run_name = args.run_name if args.run_name else f"{args.model}_{args.dataset}_seed{args.seed}"
    
    if args.wandb_mode != 'disabled':
        wandb.init(project="lanca-reboot", name=run_name, config=config_dict, mode=args.wandb_mode)
    
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if "candle" in args.dataset and config_dict.get('recon_loss_type') == 'bce':
        print("WARNING: Dataset is CANDLE but loss is BCE. Switching to MSE.")
        config_dict['recon_loss_type'] = 'mse'

    model, config_obj = get_model(args.model, config_dict)

    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    
    try:
        train_ds = CausalImageDataset(args.data_root, args.dataset, train=True, transform=transform)
        val_ds = CausalImageDataset(args.data_root, args.dataset, train=False, transform=transform)
        val_ds.set_label_stats(train_ds.label_mean, train_ds.label_std)
        if len(train_ds) > 0: _ = train_ds[0]
    except Exception as e:
        print(f"\nCRITICAL DATA ERROR: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    save_path = args.save_dir if args.run_name and "seed" in args.save_dir else os.path.join(args.save_dir, run_name)
    
    early_stopping_patience = args.scheduler_patience * 3
    
    trainer = Trainer(model, config_obj, train_loader, val_loader, device, save_path)
    trainer.fit(num_epochs=args.epochs, patience=early_stopping_patience, save_img_freq=10)

if __name__ == "__main__":
    main()