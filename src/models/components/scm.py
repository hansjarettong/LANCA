import torch
import torch.nn as nn

class SCMLayer(nn.Module):
    """
    Universal Function Approximator for SCMs (MLP-based).
    Supports different C^3 differentiable activations.
    """
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.num_causal_vars
        
        # Configurable Architecture
        hidden_dim = getattr(config, 'scm_hidden_dim', 64)
        num_layers = getattr(config, 'scm_layers', 2)
        activation_name = getattr(config, 'scm_activation', 'tanh')
        
        # Activation Registry
        activations = {
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),         # Swish (Smooth ReLU)
            'gelu': nn.GELU(),         # Gaussian Error Linear Unit
            'softplus': nn.Softplus(), # Smooth ReLU
            'relu': nn.ReLU()          # Technically not C^3, but often used as baseline
        }
        
        act_layer = activations.get(activation_name.lower(), nn.Tanh())
        if activation_name.lower() not in activations:
            print(f"[SCM] Warning: Unknown activation '{activation_name}'. Defaulting to Tanh.")

        layers = []
        
        # Input Layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(act_layer)
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_layer)
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, z_parents):
        return self.net(z_parents)