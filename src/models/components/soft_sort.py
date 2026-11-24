import torch
import torch.nn as nn
from torch import Tensor

class SoftSort_p1(nn.Module):
    """
    Differentiable sorting network using SoftSort with P=1 (absolute difference).
    Based on 'Differentiable DAG Sampling'.
    """
    def __init__(self, tau=1.0, hard=False):
        super(SoftSort_p1, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: (batch_size, n) - The values to sort (e.g., permutation weights)
        Returns: P_hat (batch_size, n, n) - The permutation matrix
        """
        scores = scores.unsqueeze(-1) # (B, n, 1)
        
        # Sort scores descending to get target positions
        sorted_scores = scores.sort(descending=True, dim=1)[0]
        
        # Compute pairwise differences
        # (B, n, 1) - (B, 1, n) -> (B, n, n)
        pairwise_diff = (scores.transpose(1, 2) - sorted_scores).abs().neg() / self.tau
        
        # Soft permutation matrix
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            # Straight-Through Estimator:
            # Convert soft P_hat to hard permutation P, but keep gradients of P_hat
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
            
        return P_hat