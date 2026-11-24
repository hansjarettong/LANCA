import numpy as np
import torch

def get_true_dag(dataset_name):
    """
    Returns the adjacency matrix for known datasets.
    """
    if "pendulum" in dataset_name.lower():
        # 0:Angle, 1:Light -> 2:ShadowLen, 3:ShadowPos
        A_true = np.zeros((4, 4))
        A_true[0, 2] = 1
        A_true[0, 3] = 1
        A_true[1, 2] = 1
        A_true[1, 3] = 1
        return A_true
    elif "flow" in dataset_name.lower():
         # 0:BallR -> 1:WaterHeight -> 3:WaterFlow
         # 2:Hole -> 3:WaterFlow
         A_true = np.zeros((4, 4))
         A_true[0, 1] = 1
         A_true[1, 3] = 1
         A_true[2, 3] = 1
         return A_true
    elif "candle" in dataset_name.lower():
        # Standard benchmarks assume independence between factors 
        # unless confounding is explicitly modeled.
        return np.zeros((6, 6))
    else:
        return None

def compute_shd(A_learned, A_true, threshold=0.1):
    """
    Computes Structural Hamming Distance (SHD).
    Counts: Extra Edges + Missing Edges + Reversed Edges.
    Reversals count as 1 error.
    """
    if A_true is None:
        return float('nan')
        
    # Binarize learned graph
    P = (np.abs(A_learned) > threshold).astype(int)
    T = (np.abs(A_true) > 0.1).astype(int)
    
    np.fill_diagonal(P, 0)
    np.fill_diagonal(T, 0)
    
    # Difference matrix
    diff = P - T
    
    extra = (diff == 1)
    missing = (diff == -1)
    
    # Find reversals: P has i->j (extra), T has j->i (missing)
    reversals = extra & missing.T
    num_reversals = reversals.sum() 
    
    # Remove reversals from extra/missing to avoid double counting
    # A reversal replaces 1 extra and 1 missing with 1 reversal error.
    num_extra = extra.sum() - num_reversals
    num_missing = missing.sum() - num_reversals
    
    return num_extra + num_missing + num_reversals

def compute_path_matrix(A):
    """
    Computes reachability (transitive closure) matrix using matrix multiplication.
    Returns M where M[i,j] = 1 if there is a path from i to j.
    """
    n = A.shape[0]
    M = np.eye(n) + np.abs(A) # adjacency + self-loops
    M = (M > 0).astype(int)
    
    # Repeated squaring to find transitive closure
    for _ in range(int(np.ceil(np.log2(n))) + 1):
        M = np.dot(M, M)
        M = (M > 0).astype(int)
        
    return M

def compute_sid(A_learned, A_true, threshold=0.1):
    """
    Computes Structural Intervention Distance (SID) proxy.
    Calculates the Hamming distance between the transitive closures of the graphs.
    """
    if A_true is None:
        return float('nan')

    G_true = (np.abs(A_true) > 0.1).astype(int)
    G_pred = (np.abs(A_learned) > threshold).astype(int)
    
    # Compute Transitive Closures
    TC_true = compute_path_matrix(G_true)
    TC_pred = compute_path_matrix(G_pred)
    
    # Remove diagonals (self-reachability) to focus on causal links
    np.fill_diagonal(TC_true, 0)
    np.fill_diagonal(TC_pred, 0)
    
    # Hamming distance between transitive closures
    return np.sum(TC_true != TC_pred)