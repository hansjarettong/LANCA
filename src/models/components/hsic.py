import torch

def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the RBF (Gaussian) kernel between two tensors.
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # Ensure tensors are 2D (B, D)
    if X.dim() == 1: X = X.unsqueeze(-1)
    if Y.dim() == 1: Y = Y.unsqueeze(-1)
        
    # Pairwise squared distances: ||x - y||^2 = x^2 - 2xy + y^2
    X_norm = (X ** 2).sum(1, keepdim=True)
    Y_norm = (Y ** 2).sum(1, keepdim=True)
    
    # K(i, j)
    K = X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)
    
    # RBF
    return torch.exp(-K / (2 * sigma ** 2))

def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """
    Centers a kernel matrix K: K_c = H * K * H
    where H = I - 1/n * 11^T
    """
    n = K.shape[0]
    I = torch.eye(n, device=K.device)
    ones = torch.ones(n, n, device=K.device) / n
    H = I - ones
    return torch.mm(H, torch.mm(K, H))

def hsic_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC).
    HSIC(X, Y) = 1/(n-1)^2 * tr(K_c * L_c)
    """
    n = X.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=X.device)

    K = rbf_kernel(X, X, sigma)
    L = rbf_kernel(Y, Y, sigma)
    
    K_c = center_kernel_matrix(K)
    L_c = center_kernel_matrix(L)
    
    # Biased estimator (simpler/stable for loss): 1/n^2 * tr(K_c L_c)
    hsic_val = torch.trace(torch.mm(K_c, L_c)) / (n ** 2)
    return hsic_val