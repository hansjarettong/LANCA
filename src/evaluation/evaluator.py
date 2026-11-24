# src/evaluation/evaluator.py

import os
import json
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import DataLoader

from .metrics import compute_mig, compute_dci, compute_sap
from .causal import get_true_dag, compute_shd, compute_sid
from .causal_metrics import load_or_train_classifiers, compute_uc, compute_cg

class Evaluator:
    def __init__(self, model, device, test_loader, save_dir):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.classifier = None

    def collect_representations(self):
        self.model.eval()
        all_z, all_labels = [], []
        loss_accum = {}
        
        with torch.no_grad():
            for batch in self.test_loader:
                mask = None
                if isinstance(batch, list):
                    if len(batch) == 3:
                        x, labels, mask = batch
                        if mask is not None: mask = mask.to(self.device)
                    elif len(batch) == 2:
                        x, labels = batch
                    else:
                        x = batch[0]
                        labels = torch.zeros(x.size(0), 1)
                else:
                    x = batch[0]
                    labels = torch.zeros(x.size(0), 1)

                x, labels = x.to(self.device), labels.to(self.device)
                
                fwd_kwargs = {}
                if "labels" in self.model.forward.__code__.co_varnames:
                    fwd_kwargs['labels'] = labels
                if "mask" in self.model.forward.__code__.co_varnames and mask is not None:
                    fwd_kwargs['mask'] = mask
                
                fwd_dict = self.model(x, **fwd_kwargs)
                
                z = fwd_dict.get('z')
                if z is None:
                    z = fwd_dict.get('z_sampled', fwd_dict.get('z_sample'))
                
                if z.dim() > 2: 
                    z = z.view(z.size(0), -1)
                
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                try:
                    losses = self.model.compute_loss(x, fwd_dict)
                    for k, v in losses.items():
                        loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
                except Exception:
                    pass
                    
        avg_losses = {f"eval_{k}": v / len(self.test_loader) for k, v in loss_accum.items() if len(self.test_loader) > 0}
        return np.concatenate(all_z), np.concatenate(all_labels), avg_losses

    def align_graph(self, latents, labels, A_learned, return_mmi=False):
        num_latents = latents.shape[1]
        num_factors = labels.shape[1]
        
        score_matrix = np.zeros((num_latents, num_factors))
        for i in range(num_latents):
            mi = mutual_info_regression(labels, latents[:, i], discrete_features=False, random_state=42)
            score_matrix[i, :] = mi

        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
        mmi_score = float(np.mean(score_matrix[row_ind, col_ind])) if len(row_ind) > 0 else 0.0
        self.last_mmi_score = mmi_score

        # Map matched latents into factor x factor adjacency matrix
        # Works for both num_latents >= num_factors and under-specified num_latents < num_factors
        A_aligned = np.zeros((num_factors, num_factors))
        for r1, c1 in zip(row_ind, col_ind):
            for r2, c2 in zip(row_ind, col_ind):
                if r1 < A_learned.shape[0] and r2 < A_learned.shape[1]:
                    A_aligned[c1, c2] = A_learned[r1, r2]

        if return_mmi:
            return A_aligned, mmi_score
        return A_aligned

    def align_graph_alternatives(self, latents, labels, A_learned, dataset_name=None):
        """
        Evaluates 4 graph alignment strategies (R2.1.8):
        1. hungarian_mi: 1-to-1 Hungarian matching on Mutual Information matrix
        2. greedy_mi: many-to-1 argmax MI per factor
        3. hungarian_spearman: 1-to-1 Hungarian matching on Spearman rank correlation
        4. hungarian_dci: 1-to-1 Hungarian matching on Gradient Boosting feature importance
        """
        num_latents = latents.shape[1]
        num_factors = labels.shape[1]
        A_true = get_true_dag(dataset_name) if dataset_name else None

        # 1. MI Score Matrix
        mi_matrix = np.zeros((num_latents, num_factors))
        for i in range(num_latents):
            mi_matrix[i, :] = mutual_info_regression(labels, latents[:, i], discrete_features=False, random_state=42)

        # 2. Spearman Rank Correlation Matrix
        spearman_matrix = np.zeros((num_latents, num_factors))
        for i in range(num_latents):
            for j in range(num_factors):
                corr, _ = spearmanr(latents[:, i], labels[:, j])
                spearman_matrix[i, j] = 0.0 if np.isnan(corr) else abs(corr)

        # 3. DCI / GBDT Feature Importance Matrix
        dci_matrix = np.zeros((num_latents, num_factors))
        for j in range(num_factors):
            gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            gbr.fit(latents, labels[:, j])
            dci_matrix[:, j] = gbr.feature_importances_

        strategies = {}

        # 1. Hungarian MI
        r_mi, c_mi = linear_sum_assignment(mi_matrix, maximize=True)
        # Factor j mapped to which latent
        map_h_mi = {c: r for r, c in zip(r_mi, c_mi)}
        A_h_mi = np.zeros((num_factors, num_factors))
        for r1, c1 in zip(r_mi, c_mi):
            for r2, c2 in zip(r_mi, c_mi):
                if r1 < A_learned.shape[0] and r2 < A_learned.shape[1]:
                    A_h_mi[c1, c2] = A_learned[r1, r2]
        strategies['hungarian_mi'] = {
            'mapping': [map_h_mi.get(j, -1) for j in range(num_factors)],
            'mmi_score': float(np.mean(mi_matrix[r_mi, c_mi])),
            'A_aligned': A_h_mi
        }

        # 2. Greedy MI (argmax per factor)
        map_greedy_mi = [int(np.argmax(mi_matrix[:, j])) for j in range(num_factors)]
        A_greedy = np.zeros((num_factors, num_factors))
        for j1 in range(num_factors):
            for j2 in range(num_factors):
                u, v = map_greedy_mi[j1], map_greedy_mi[j2]
                if u < A_learned.shape[0] and v < A_learned.shape[1]:
                    A_greedy[j1, j2] = A_learned[u, v]
        strategies['greedy_mi'] = {
            'mapping': map_greedy_mi,
            'mmi_score': float(np.mean([mi_matrix[map_greedy_mi[j], j] for j in range(num_factors)])),
            'A_aligned': A_greedy
        }

        # 3. Hungarian Spearman
        r_sp, c_sp = linear_sum_assignment(spearman_matrix, maximize=True)
        map_sp = {c: r for r, c in zip(r_sp, c_sp)}
        A_sp = np.zeros((num_factors, num_factors))
        for r1, c1 in zip(r_sp, c_sp):
            for r2, c2 in zip(r_sp, c_sp):
                if r1 < A_learned.shape[0] and r2 < A_learned.shape[1]:
                    A_sp[c1, c2] = A_learned[r1, r2]
        strategies['hungarian_spearman'] = {
            'mapping': [map_sp.get(j, -1) for j in range(num_factors)],
            'mean_corr': float(np.mean(spearman_matrix[r_sp, c_sp])),
            'A_aligned': A_sp
        }

        # 4. Hungarian DCI
        r_dci, c_dci = linear_sum_assignment(dci_matrix, maximize=True)
        map_dci = {c: r for r, c in zip(r_dci, c_dci)}
        A_dci = np.zeros((num_factors, num_factors))
        for r1, c1 in zip(r_dci, c_dci):
            for r2, c2 in zip(r_dci, c_dci):
                if r1 < A_learned.shape[0] and r2 < A_learned.shape[1]:
                    A_dci[c1, c2] = A_learned[r1, r2]
        strategies['hungarian_dci'] = {
            'mapping': [map_dci.get(j, -1) for j in range(num_factors)],
            'mean_importance': float(np.mean(dci_matrix[r_dci, c_dci])),
            'A_aligned': A_dci
        }

        ref_mapping = strategies['hungarian_mi']['mapping']
        for strat_name, strat_data in strategies.items():
            overlap = sum([1 for m1, m2 in zip(ref_mapping, strat_data['mapping']) if m1 == m2]) / float(num_factors)
            strat_data['overlap_with_hungarian_mi'] = overlap
            if A_true is not None:
                strat_data['shd'] = compute_shd(strat_data['A_aligned'], A_true)
                strat_data['sid'] = compute_sid(strat_data['A_aligned'], A_true)

        return strategies

    def _make_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        return obj

    def run(self, dataset_name):
        print(f"--- Evaluating on {dataset_name} ---")
        os.makedirs(self.save_dir, exist_ok=True)

        z, u, test_losses = self.collect_representations()
        results = {}
        results.update(test_losses)
        
        if u.shape[1] > 1:
            print("Computing Disentanglement Metrics...")
            try:
                results['mig'] = compute_mig(z, u)
                results.update(compute_dci(z, u))
                results.update(compute_sap(z, u))
            except Exception as e:
                print(f"Error disentanglement metrics: {e}")

            # Compute MMI score (mean mutual information of Hungarian-matched pairs)
            try:
                num_lat = z.shape[1]
                num_fac = u.shape[1]
                score_mat = np.zeros((num_lat, num_fac))
                for i in range(num_lat):
                    score_mat[i, :] = mutual_info_regression(u, z[:, i], discrete_features=False, random_state=42)
                r_idx, c_idx = linear_sum_assignment(score_mat, maximize=True)
                results['mmi_score'] = float(np.mean(score_mat[r_idx, c_idx])) if len(r_idx) > 0 else 0.0
            except Exception as e:
                print(f"Error computing mmi_score: {e}")

            try:
                if hasattr(self.test_loader.dataset, 'root'):
                    data_root = self.test_loader.dataset.root
                else:
                    data_root = os.path.join("data", dataset_name)
                
                classifier_dir = os.path.join(os.path.dirname(data_root), "classifiers", dataset_name)
                
                print(f"Loading/Training Ensemble Classifiers in: {classifier_dir}")
                
                x_dummy = next(iter(self.test_loader))[0] if len(self.test_loader) > 0 else None
                if x_dummy is not None:
                    img_channels = x_dummy.shape[1]
                    num_factors = u.shape[1]

                    self.classifier = load_or_train_classifiers(
                        self.test_loader, 
                        num_factors, 
                        img_channels, 
                        self.device, 
                        classifier_dir,
                        num_models=10
                    )
                    
                    rhos = [1, 2, 5, 7]
                    for rho in rhos:
                        print(f"Computing UC/CG for rho={rho}...")
                        uc_res, factor_map = compute_uc(z, u, rho=rho)
                        results.update(uc_res)
                        
                        cg_res = compute_cg(
                            self.model, 
                            self.classifier, 
                            self.test_loader, 
                            factor_map, 
                            self.device,
                            total_latents=z.shape[1] 
                        )
                        results[f"CG_score_rho{rho}"] = cg_res["CG_score"]
            except Exception as e:
                print(f"Error causal metrics: {e}")
                import traceback
                traceback.print_exc()
        
        has_scm = hasattr(self.model, "scm")
        has_dag_layer = hasattr(self.model, "dag_layer")
        has_A = hasattr(self.model, "A")
        
        if has_scm or has_dag_layer or has_A:
            print("Computing Graph Metrics (SHD/SID)...")
            try:
                A_struct = None
                causal_order = None
                if hasattr(self.model, "get_functional_graph"):
                    A_learned, causal_order = self.model.get_functional_graph(threshold=1e-3)
                    if hasattr(self.model.scm, "dag_layer"):
                        A_struct, _ = self.model.scm.dag_layer.get_dag_and_order(hard=True)
                elif has_scm: 
                    A_learned, causal_order = self.model.scm.dag_layer.get_dag_and_order(hard=True)
                    A_struct = A_learned
                elif has_dag_layer: 
                    A_learned = self.model.dag_layer.A
                    A_struct = A_learned
                else: 
                    A_learned = self.model.A
                    A_struct = A_learned
                
                if torch.is_tensor(A_learned):
                    A_np = A_learned.detach().cpu().numpy()
                else:
                    A_np = np.array(A_learned)

                if A_struct is not None:
                    A_struct_np = A_struct.detach().cpu().numpy() if torch.is_tensor(A_struct) else np.array(A_struct)
                    np.savetxt(os.path.join(self.save_dir, "adj_matrix_struct.csv"), A_struct_np, delimiter=",")
                
                np.savetxt(os.path.join(self.save_dir, "adj_matrix.csv"), A_np, delimiter=",")
                
                if causal_order is not None:
                    if torch.is_tensor(causal_order):
                        results['causal_order'] = causal_order.tolist()
                    else:
                        results['causal_order'] = list(causal_order)

                if u.shape[1] > 1:
                    A_aligned, mmi_score = self.align_graph(z, u, A_np, return_mmi=True)
                    results['mmi_score'] = mmi_score
                    A_true = get_true_dag(dataset_name)
                    
                    if A_true is not None:
                        shd = compute_shd(A_aligned, A_true)
                        sid = compute_sid(A_aligned, A_true)
                        
                        results['shd'] = shd
                        results['sid'] = sid
                        print(f"SHD: {shd}, SID: {sid}, MMI: {mmi_score:.4f}")
                    else:
                        print("True DAG not defined. Skipping SHD/SID.")
                        
            except Exception as e:
                print(f"Error computing graph metrics: {e}")

        results = self._make_serializable(results)
        results_path = os.path.join(self.save_dir, "metrics.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
        return results