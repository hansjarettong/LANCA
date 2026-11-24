import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import entropy

def get_latents_and_labels(model: torch.nn.Module, 
                           dataloader: DataLoader, 
                           device: str,
                           model_name: str = "lanca") -> tuple:
    """
    Runs the model over the entire dataloader to collect all
    learned latent variables (z) and ground-truth labels (u).
    
    Adapted for LANCA project structure.
    """
    model.eval()
    model.to(device)
    
    all_z_list = []
    all_labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Matches logic in Evaluator/Trainer
            mask = None
            if isinstance(batch, list):
                if len(batch) == 3:
                    images, labels, mask = batch
                elif len(batch) == 2:
                    images, labels = batch
                else:
                    images = batch[0]
                    # Create dummy labels if missing
                    labels = torch.zeros(images.size(0), 1)
            else:
                images = batch
                labels = torch.zeros(images.size(0), 1)

            images = images.to(device)
            labels = labels.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            # --- Forward Pass ---
            # CausalVAE and LANCA accept labels/masks, others might simply ignore extra args
            # We check the signature to be safe, or just pass kwargs if model allows
            fwd_kwargs = {}
            
            # Introspect forward method
            forward_args = model.forward.__code__.co_varnames
            if "labels" in forward_args:
                fwd_kwargs['labels'] = labels
            if "mask" in forward_args and mask is not None:
                fwd_kwargs['mask'] = mask
                
            fwd_dict = model(images, **fwd_kwargs)

            # --- Extract Latents (z) ---
            # Your current models (LANCA, CausalVAE, BetaVAE) all return "z" in the dict
            if "z" in fwd_dict:
                z = fwd_dict["z"]
            elif "z_sample" in fwd_dict:
                z = fwd_dict["z_sample"]
            elif "z_sampled" in fwd_dict:
                z = fwd_dict["z_sampled"]
            elif "z_samples" in fwd_dict:
                z = fwd_dict["z_samples"][0]
            else:
                # Fallback: try to find any key that looks like latents
                keys = list(fwd_dict.keys())
                raise KeyError(f"Could not find latent key 'z' in model output. Available keys: {keys}")
            
            # Ensure z is 2D (Batch, Dim)
            if z.dim() > 2:
                z = z.reshape(z.shape[0], -1)
                
            all_z_list.append(z.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())

    all_latents_z = np.concatenate(all_z_list, axis=0)
    all_labels_u = np.concatenate(all_labels_list, axis=0)
    
    return all_latents_z, all_labels_u

def compute_mig(latents, labels):
    """
    Mutual Information Gap (MIG).
    Normalized by factor entropy.
    """
    num_factors = labels.shape[1]
    num_latents = latents.shape[1]
    
    mi_matrix = np.zeros((num_factors, num_latents))
    for i in range(num_factors):
        # Ensure labels are 1D
        y = labels[:, i].ravel()
        # random_state ensures reproducibility
        mi_matrix[i] = mutual_info_regression(latents, y, discrete_features=False, random_state=42)
        
    # Sort MI for each factor
    sorted_mi = np.sort(mi_matrix, axis=1)[:, ::-1]
    
    if num_latents < 2:
        return 0.0
        
    # Normalize by entropy of the factor (H(y))
    # We estimate H(y) using a histogram
    entropy_factors = np.zeros(num_factors)
    for i in range(num_factors):
        y = labels[:, i]
        # Heuristic: if continuous, bin it. If discrete, counts.
        # Simple unified approach: histogram with auto bins
        counts, _ = np.histogram(y, bins='auto')
        # Convert counts to probs
        probs = counts / np.sum(counts)
        entropy_factors[i] = entropy(probs)
        
    # Avoid div by zero
    entropy_factors[entropy_factors == 0] = 1.0
    
    gaps = (sorted_mi[:, 0] - sorted_mi[:, 1]) / entropy_factors
    return np.mean(gaps)

def compute_dci(latents, labels):
    """
    Disentanglement, Completeness, Informativeness (DCI).
    Handles Discrete vs Continuous factors automatically.
    """
    train_ratio = 0.8
    split = int(len(latents) * train_ratio)
    
    z_train, z_test = latents[:split], latents[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    num_factors = labels.shape[1]
    num_latents = latents.shape[1]
    
    importance_matrix = np.zeros((num_latents, num_factors))
    test_errors = []
    
    for i in range(num_factors):
        # Heuristic: Check if factor is discrete
        unique_vals = np.unique(y_train[:, i])
        is_discrete = len(unique_vals) < 20 and np.all(unique_vals % 1 == 0)
        
        if is_discrete:
            # Use Classifier
            model = GradientBoostingClassifier(random_state=42)
            model.fit(z_train, y_train[:, i].astype(int))
            score = model.score(z_test, y_test[:, i].astype(int)) # Accuracy
        else:
            # Use Regressor
            model = GradientBoostingRegressor(random_state=42)
            model.fit(z_train, y_train[:, i])
            score = model.score(z_test, y_test[:, i]) # R^2 score
            # R^2 can be negative, clamp to 0 for 'informativeness' concept
            score = max(0.0, score)
            
        test_errors.append(score)
        importance_matrix[:, i] = np.abs(model.feature_importances_)

    # Disentanglement (Entropy of rows)
    row_sums = importance_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = importance_matrix / row_sums
    
    # Entropy base num_factors
    ent = entropy(probs.T + 1e-10, base=num_factors)
    disentanglement = 1.0 - ent
    
    # Weighted average
    code_importance = importance_matrix.sum(axis=1)
    if code_importance.sum() == 0:
        avg_disentanglement = 0.0
    else:
        avg_disentanglement = np.sum(disentanglement * code_importance) / code_importance.sum()

    return {
        "dci_disentanglement": avg_disentanglement,
        "dci_informativeness": np.mean(test_errors)
    }

def compute_sap(latents, labels):
    """
    Separated Attribute Predictability (SAP).
    """
    train_ratio = 0.8
    split = int(len(latents) * train_ratio)
    
    z_train, z_test = latents[:split], latents[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    num_factors = labels.shape[1]
    num_latents = latents.shape[1]
    
    score_matrix = np.zeros((num_latents, num_factors))
    
    for i in range(num_latents):
        for j in range(num_factors):
            # Reshape for sklearn
            feat_train = z_train[:, i].reshape(-1, 1)
            feat_test = z_test[:, i].reshape(-1, 1)
            
            # Heuristic for Discrete/Continuous
            unique_vals = np.unique(y_train[:, j])
            is_discrete = len(unique_vals) < 20 and np.all(unique_vals % 1 == 0)
            
            if is_discrete:
                model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42)
                model.fit(feat_train, y_train[:, j].astype(int))
                score = model.score(feat_test, y_test[:, j].astype(int)) # Accuracy
            else:
                model = LinearRegression()
                model.fit(feat_train, y_train[:, j])
                score = max(0, model.score(feat_test, y_test[:, j])) # R^2
            
            score_matrix[i, j] = score
            
    # Gap between top 2 latents for each factor
    sorted_scores = np.sort(score_matrix, axis=0)[::-1] # Sort columns descending
    
    if num_latents < 2:
        return {"sap_score": 0.0}
        
    gaps = sorted_scores[0, :] - sorted_scores[1, :]
    return {"sap_score": np.mean(gaps)}