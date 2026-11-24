# src/evaluation/causal_metrics.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import List

# --- 1. Factor Classifier Architecture ---
class FactorClassifier(nn.Module):
    def __init__(self, img_channels: int, num_factors: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 16, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 4, 1), nn.ReLU(True), 
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64), 
            nn.ReLU(True),
            nn.Linear(64, num_factors)
        )
        self.register_buffer('lbl_min', torch.zeros(num_factors))
        self.register_buffer('lbl_range', torch.ones(num_factors))

    def forward(self, x):
        return self.net(x)

# --- 2. Ensemble Manager ---
class FactorClassifierEnsemble:
    def __init__(self, models: List[nn.Module]):
        self.models = models
        
    def eval(self):
        for m in self.models: m.eval()
        
    def to(self, device):
        for m in self.models: m.to(device)
        return self
        
    def __call__(self, x):
        preds = []
        with torch.no_grad():
            for m in self.models:
                preds.append(m(x))
        return torch.stack(preds).mean(dim=0)

def load_or_train_classifiers(dataloader, num_factors, img_channels, device, save_dir, num_models=10):
    os.makedirs(save_dir, exist_ok=True)
    models = []
    
    all_labels = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)): 
            labels = batch[1]
        else: 
            labels = None
        
        if labels is not None:
            all_labels.append(labels)
            
    if not all_labels:
        raise ValueError("Dataloader provided no labels.")
        
    all_labels_cat = torch.cat(all_labels, dim=0)
    lbl_min = all_labels_cat.min(dim=0)[0].to(device)
    lbl_max = all_labels_cat.max(dim=0)[0].to(device)
    lbl_range = lbl_max - lbl_min
    lbl_range[lbl_range == 0] = 1.0

    print(f"Ensemble Strategy: Using {num_models} classifiers.")
    
    for i in range(num_models):
        path = os.path.join(save_dir, f"classifier_{i}.pth")
        model = FactorClassifier(img_channels, num_factors).to(device)
        model.lbl_min = lbl_min
        model.lbl_range = lbl_range
        
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"Training factor classifier {i+1}/{num_models}...")
            _train_single_classifier(model, dataloader, device)
            torch.save(model.state_dict(), path)
            
        model.eval()
        models.append(model)
        
    return FactorClassifierEnsemble(models)

def _train_single_classifier(classifier, dataloader, device, num_epochs=5):
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    classifier.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)): 
                images, labels = batch[0], batch[1]
            else: continue
            
            images, labels = images.to(device), labels.to(device)
            labels_norm = (labels - classifier.lbl_min) / classifier.lbl_range
            
            optimizer.zero_grad()
            preds = classifier(images)
            loss = criterion(preds, labels_norm)
            loss.backward()
            optimizer.step()

# --- 3. IRS and UC Metrics ---

def discretize_factors(factors: np.ndarray, num_bins: int = 20) -> np.ndarray:
    factors_discrete = np.zeros_like(factors, dtype=int)
    for k in range(factors.shape[1]):
        factor_col = factors[:, k]
        if np.max(factor_col) == np.min(factor_col):
            factors_discrete[:, k] = 0
            continue
        _, edges = np.histogram(factor_col, bins=num_bins)
        idxs = np.digitize(factor_col, edges[:-1]) - 1
        factors_discrete[:, k] = idxs
    return factors_discrete

def compute_irs_matrix(latents: np.ndarray, factors: np.ndarray, diff_quantile: float = 0.99, num_bins: int = 20):
    num_latents = latents.shape[1]
    factors_discrete = discretize_factors(factors, num_bins=num_bins)
    num_factors = factors_discrete.shape[1]
    
    global_means = np.mean(latents, axis=0)
    max_deviations = np.max(np.abs(latents - global_means), axis=0)
    max_deviations[max_deviations == 0] = 1.0 
    
    cum_deviations = np.zeros((num_latents, num_factors))
    
    for k in range(num_factors):
        unique_vals = np.unique(factors_discrete[:, k])
        for val in unique_vals:
            mask = (factors_discrete[:, k] == val)
            if not np.any(mask): continue
            
            subset_latents = latents[mask]
            e_loc = np.mean(subset_latents, axis=0)
            
            diffs = np.abs(subset_latents - e_loc)
            current_max_diffs = np.percentile(diffs, q=diff_quantile * 100, axis=0)
            cum_deviations[:, k] += current_max_diffs
            
        cum_deviations[:, k] /= len(unique_vals)
        
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations
    return irs_matrix, max_deviations

def compute_uc(latents: np.ndarray, labels: np.ndarray, rho: int = 1):
    irs_matrix, max_deviations = compute_irs_matrix(latents, labels)
    
    disentanglement_scores = irs_matrix.max(axis=1)
    if np.sum(max_deviations) > 0:
        irs_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        irs_score = np.mean(disentanglement_scores)
        
    num_factors = irs_matrix.shape[1]
    factor_latent_map = {}
    
    for j in range(num_factors):
        scores = irs_matrix[:, j]
        top_rho_indices = np.argsort(scores)[-rho:] 
        factor_latent_map[j] = set(top_rho_indices.tolist())

    total_jaccard = 0.0
    num_pairs = 0
    for i in range(num_factors):
        for j in range(i + 1, num_factors):
            set_i = factor_latent_map[i]
            set_j = factor_latent_map[j]
            union = len(set_i.union(set_j))
            if union > 0:
                jaccard = len(set_i.intersection(set_j)) / union
            else:
                jaccard = 0.0
            total_jaccard += jaccard
            num_pairs += 1
            
    uc_score = 1.0 - (total_jaccard / num_pairs if num_pairs > 0 else 0.0)
    
    return {
        f"UC_score_rho{rho}": uc_score,
        "IRS_score": irs_score
    }, factor_latent_map

# --- 4. Updated CG Metric ---

def get_global_ranges(model, dataloader, device):
    """Computes global min and max for latents across the dataset."""
    model.eval()
    all_z = []
    
    def extract_z(out):
        if isinstance(out, dict):
            return out.get('z', out.get('z_sampled', out.get('z_sample')))
        return out

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)): x = batch[0]
            else: x = batch
            x = x.to(device)
            z = extract_z(model(x))
            if z.dim() > 2: z = z.view(z.size(0), -1)
            all_z.append(z.cpu())
            
    all_z = torch.cat(all_z, dim=0)
    g_min = all_z.min(dim=0)[0].to(device)
    g_max = all_z.max(dim=0)[0].to(device)
    return g_min, g_max

def get_max_deviations_global(target_dims, latents, g_min, g_max):
    intervened_latents = latents.clone()
    if not target_dims: return intervened_latents
    
    for dim in target_dims:
        val = latents[:, dim]
        d_min = torch.abs(val - g_min[dim])
        d_max = torch.abs(val - g_max[dim])
        target_val = torch.where(d_max > d_min, g_max[dim], g_min[dim])
        intervened_latents[:, dim] = target_val
        
    return intervened_latents

@torch.no_grad()
def compute_cg(model, classifier, dataloader, factor_latent_map, device, total_latents=None):
    """
    Computes Counterfactual Generativeness (CG).
    Now accepts total_latents to handle unsupervised latent counts correctly.
    """
    model.eval()
    
    # 1. Precompute Global Ranges
    g_min, g_max = get_global_ranges(model, dataloader, device)
    
    # If total_latents not provided, infer from data
    if total_latents is None:
        total_latents = g_min.shape[0]
        
    num_factors = len(factor_latent_map)
    global_accumulators = {i: {'I': 0.0, 'notI': 0.0} for i in range(num_factors)}
    total_samples = 0

    def encode(x):
        if hasattr(model, 'encoder_base'):
            h = model.encoder_base(x)
            if hasattr(model, 'fc_z'): return model.fc_z(h) 
            if hasattr(model, 'fc_mu'): return model.fc_mu(h)
            if hasattr(model, 'fc_enc_mu'): return model.fc_enc_mu(h)
        res = model(x)
        if isinstance(res, dict):
            return res.get('z', res.get('z_sampled'))
        return res

    def decode(z):
        bridge = getattr(model, 'fc_decode_in', getattr(model, 'fc_decode', None))
        if bridge:
            h = bridge(z)
            if hasattr(model, 'decoder_base'):
                return torch.sigmoid(model.decoder_base(h))
        res = model(z)
        if isinstance(res, dict):
            return torch.sigmoid(res.get('x_recon_logits', res.get('x_recon')))
        return torch.sigmoid(res)

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            continue
            
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        
        z_batch = encode(images)
        if z_batch.dim() > 2: z_batch = z_batch.view(batch_size, -1)
        
        # Base Prediction
        img_base = decode(z_batch)
        logits_base = classifier(img_base)
        
        for i in range(num_factors):
            Z_I_indices = list(factor_latent_map.get(i, set()))
            # Correctly identify irrelevant latents using total_latents
            Z_notI_indices = list(set(range(total_latents)) - set(Z_I_indices))
            
            pred_base = logits_base
            target_col = i
            
            # Intervention 1: Change Z_I
            if Z_I_indices:
                z_int_I = get_max_deviations_global(Z_I_indices, z_batch, g_min, g_max)
                img_I = decode(z_int_I)
                pred_I = classifier(img_I)
                effect_I_sum = torch.abs(pred_base[:, target_col] - pred_I[:, target_col]).sum().item()
            else:
                effect_I_sum = 0.0

            # Intervention 2: Change Z_notI
            if Z_notI_indices:
                z_int_notI = get_max_deviations_global(Z_notI_indices, z_batch, g_min, g_max)
                img_notI = decode(z_int_notI)
                pred_notI = classifier(img_notI)
                effect_notI_sum = torch.abs(pred_base[:, target_col] - pred_notI[:, target_col]).sum().item()
            else:
                effect_notI_sum = 0.0

            global_accumulators[i]['I'] += effect_I_sum
            global_accumulators[i]['notI'] += effect_notI_sum

    if total_samples == 0: return {"CG_score": 0.0}

    final_diffs = []
    for i in range(num_factors):
        ace_I = global_accumulators[i]['I'] / total_samples
        ace_notI = global_accumulators[i]['notI'] / total_samples
        score = abs(ace_I - ace_notI)
        final_diffs.append(score)
        
    return {"CG_score": np.mean(final_diffs)}