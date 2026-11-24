import sys
import os
import argparse
import yaml
import torch
import json
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.evaluator import Evaluator
from src.models.lanca import LANCA, LANCAConfig
from src.models.baselines.beta_vae import BetaVAE, BetaVAEConfig
from src.models.baselines.causal_vae import CausalVAE, CausalVAEConfig
from src.models.baselines.factor_vae import FactorVAE, FactorVAEConfig
from src.utils.data_loader import CausalImageDataset

def get_model(model_name, config_dict):
    """
    Instantiates the model class based on the name and config dictionary.
    Robustly filters config_dict to only pass valid arguments to the dataclass.
    """
    def clean_cfg(cls, d):
        valid = cls.__annotations__.keys()
        return {k: v for k, v in d.items() if k in valid}

    if model_name == 'lanca':
        return LANCA(LANCAConfig(**clean_cfg(LANCAConfig, config_dict)))
    elif model_name == 'beta_vae':
        return BetaVAE(BetaVAEConfig(**clean_cfg(BetaVAEConfig, config_dict)))
    elif model_name == 'factor_vae':
        return FactorVAE(FactorVAEConfig(**clean_cfg(FactorVAEConfig, config_dict)))
    elif model_name == 'causal_vae':
        return CausalVAE(CausalVAEConfig(**clean_cfg(CausalVAEConfig, config_dict)))
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pendulum')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Note: Other args might be passed by tune.py but ignored if not used here
    # We use parse_known_args to ignore extra flags safely
    args, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Config
    config_path = os.path.join(args.run_dir, "config.json")
    
    # Fix for potential nested directory issues
    if not os.path.exists(config_path):
        parent_config = os.path.join(os.path.dirname(args.run_dir), "config.json")
        if os.path.exists(parent_config):
            config_path = parent_config
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found for {args.run_dir}")
        return

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 2. Instantiate Model
    try:
        model = get_model(args.model, config_dict)
    except Exception as e:
        print(f"Error instantiating model {args.model}: {e}")
        return

    # 3. Load Weights
    model_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 4. Prepare Data
    transform = T.Compose([T.Resize((96, 96)), T.ToTensor()])
    # Use the dataset name from config if available, else argument
    dataset_name = config_dict.get('dataset', args.dataset)
    
    try:
        test_ds = CausalImageDataset(args.data_root, dataset_name, train=False, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return
    
    # 5. Run Evaluation
    evaluator = Evaluator(model, device, test_loader, args.run_dir)
    evaluator.run(dataset_name)

if __name__ == "__main__":
    main()