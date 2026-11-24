import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_sort import SoftSort_p1

class DifferentiableDAGLayer(nn.Module):
    """
    Deterministic Differentiable DAG Layer.
    Learns permutation P and edge weights U s.t. A = P.T @ U @ P.
    """
    def __init__(self, num_nodes: int, initial_adj: torch.Tensor = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.soft_sort = SoftSort_p1(hard=True, tau=1.0)
        
        # Learnable parameters
        # Initialize with small noise to break symmetry
        self.perm_weights = nn.Parameter(torch.randn(num_nodes) * 0.01)
        
        # Edges (initially uniform)
        e = torch.zeros(num_nodes, num_nodes)
        torch.nn.init.uniform_(e, a=0.0, b=0.1) 
        
        if initial_adj is not None:
            zero_indices = (1 - initial_adj).bool()
            e.requires_grad = False
            e[zero_indices] = -300.0
            e.requires_grad = True
            
        # No self-loops
        e.requires_grad = False
        torch.diagonal(e).fill_(-300.0)
        e.requires_grad = True
        
        self.edge_log_params = nn.Parameter(e)
        self.register_buffer('mask', torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1))

        if initial_adj is not None:
            self.edge_log_params.register_hook(lambda grad: grad * initial_adj.float())


    def get_dag_and_order(self, hard=True, tau_edges=1.0):
        """
        Computes the DAG adjacency matrix A using Node-Based Logic.
        """
        # Get Permutation P
        logits = self.perm_weights
        
        # Add infinitesimal noise to break ties during hard sorting
        if hard:
             logits = logits + torch.randn_like(logits) * 1e-6

        original_hard = self.soft_sort.hard
        self.soft_sort.hard = hard
        P = self.soft_sort(logits.view(1, -1)).squeeze(0)
        self.soft_sort.hard = original_hard

        # Get Edges U with Temperature Control
        U_soft = torch.sigmoid(self.edge_log_params / tau_edges)
        
        if hard:
            U_hard = (U_soft >= 0.5).float()
            U = (U_hard - U_soft).detach() + U_soft
        else:
            U = U_soft
            
        # This transforms the upper-triangular mask M into the node-space configuration.
        mask_node = P.T @ self.mask @ P
        
        # Apply the dynamic mask to the static node parameters
        A = U * mask_node
        
        if hard:
            A = torch.clamp(A, 0, 1)

        # Extract causal order (indices)
        # P is (Rank x Node). argmax(axis=1) gives which Node is at Rank 0, Rank 1, etc.
        causal_order = P.detach().cpu().numpy().argmax(axis=1)
        
        return A, causal_order

    def sparsity_regularizer(self, prior_prob=0.01, tau_edges=1.0):
        logits = self.edge_log_params / tau_edges
        prior = torch.full_like(logits, prior_prob)
        return F.binary_cross_entropy_with_logits(logits, prior, reduction='mean')

    def permutation_regularizer(self):
        original_hard = self.soft_sort.hard
        self.soft_sort.hard = False
        P_soft = self.soft_sort(self.perm_weights.view(1, -1)).squeeze(0)
        self.soft_sort.hard = original_hard
        
        epsilon = 1e-8
        entropy = -torch.sum(P_soft * torch.log(P_soft + epsilon))
        return entropy