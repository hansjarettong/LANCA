import torch

def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel_type: str = "imq", kernel_mul: float = 2.0, kernel_n_kernels: int = 5) -> torch.Tensor:
    batch_size = x.size(0)
    n_samples = batch_size
    
    xy = torch.cat([x, y], dim=0)
    xx = torch.mm(xy, xy.t())
    yy = torch.sum(xy*xy, dim=1, keepdim=True)
    zz = torch.sum(xy*xy, dim=1).unsqueeze(0)
    dist = yy + zz - 2*xx
    
    bandwidth_range = [kernel_mul**i for i in range(kernel_n_kernels)]
    loss = torch.tensor(0.0, device=x.device)
    
    for scale in bandwidth_range:
        c = scale 
        
        if kernel_type.lower() == "rbf":
            # Gaussian / RBF Kernel
            # k(x,y) = exp( -||x-y||^2 / (2*sigma^2) )
            # Here c acts as 2*sigma^2 or bandwidth
            k_val = torch.exp(-dist / c)
            
        else:
            # Inverse Multiquadratics (IMQ) - Default/Current
            # k(x,y) = C / (C + ||x-y||^2)
            k_val = c / (c + dist)
        
        k_xx = k_val[:n_samples, :n_samples]
        k_yy = k_val[n_samples:, n_samples:]
        k_xy = k_val[:n_samples, n_samples:]
        
        loss += torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
        
    return loss