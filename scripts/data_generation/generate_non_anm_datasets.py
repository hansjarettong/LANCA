import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any, List

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.data_generation.pendulum import (
    PENDULUM_ANGLE_RANGE, LIGHT_ANGLE_RANGE, LIGHT_ANGLE_SKIP,
    PENDULUM_PIVOT, PENDULUM_LENGTH, PENDULUM_BALL_RADIUS, PENDULUM_ROD_LENGTH,
    LIGHT_SOURCE_Y, LIGHT_SOURCE_RADIUS, SHADOW_Y_POSITION, MIN_SHADOW_LENGTH,
    FIGURE_X_LIM, FIGURE_Y_LIM, NORMALIZATION_SCALE, projection, get_deterministic_metadata as get_pendulum_det,
    generate_single_image as gen_pendulum_img
)
from scripts.data_generation.flow import (
    BALL_R_RANGE, H_RAW_RANGE, HOLE_POS_RANGE, GRAVITY_CONST,
    BALL_R_SCALE, H_RAW_SCALE, HOLE_POS_SCALE, CUP_X_LEFT, CUP_X_RIGHT, CUP_WIDTH,
    CUP_TOP_Y, CUP_BOTTOM_Y, BALL_CENTER_X, BALL_Y_OFFSET, HOLE_HEIGHT, WATER_LINE_Y_MIN,
    get_deterministic_metadata as get_flow_det, generate_single_image as gen_flow_img
)

def generate_pendulum_dataset(noise_type: str, noise_ratio: float = 0.1, root_dir: str = './data'):
    dataset_name = f'pendulum_{noise_type}'
    base_path = os.path.join(root_dir, dataset_name)
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print(f"Generating Pendulum ({noise_type}) to {base_path}...")
    df_det = pd.DataFrame(get_pendulum_det())
    noise_std_shade = df_det['shade_deterministic'].std() * noise_ratio
    noise_std_mid = df_det['mid_deterministic'].std() * noise_ratio

    # Normalization statistics for parents (i, j)
    mean_i, std_i = df_det['i'].mean(), df_det['i'].std()
    mean_j, std_j = df_det['j'].mean(), df_det['j'].std()

    rng = np.random.default_rng(seed=42)
    train_rows, test_rows = [], []

    for idx, det_row in df_det.iterrows():
        i_val = det_row['i']
        j_val = det_row['j']
        
        # Parent distance from mean
        pa_dist = np.sqrt(((i_val - mean_i)/std_i)**2 + ((j_val - mean_j)/std_j)**2) / np.sqrt(2.0)

        if noise_type == 'hetero':
            # Heteroscedastic noise: parent-dependent noise scale
            scale_factor = 0.5 + pa_dist
            eps_shade = rng.normal(0.0, 1.0)
            eps_mid = rng.normal(0.0, 1.0)
            shade_noise = noise_std_shade * scale_factor * eps_shade
            mid_noise = noise_std_mid * scale_factor * eps_mid
            shade_final = max(MIN_SHADOW_LENGTH, det_row['shade_deterministic'] + shade_noise)
            mid_final = det_row['mid_deterministic'] + mid_noise

        elif noise_type == 'mult':
            # Multiplicative / Post-Nonlinear noise: s = h(pa) * exp(eta * eps)
            eps_shade = rng.normal(0.0, 1.0)
            eps_mid = rng.normal(0.0, 1.0)
            shade_final = max(MIN_SHADOW_LENGTH, det_row['shade_deterministic'] * np.exp(noise_ratio * eps_shade))
            # For mid, keep sign/shift centered around 10.0:
            mid_centered = det_row['mid_deterministic'] - 10.0
            mid_final = 10.0 + mid_centered * np.exp(noise_ratio * eps_mid)

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        fig, ax, data = gen_pendulum_img(int(i_val), int(j_val), shade_final, mid_final)
        data['filename'] = f"{idx}.png"
        for k, (min_v, range_v) in NORMALIZATION_SCALE.items():
            data[f'{k}_norm'] = (data[k] - min_v) / range_v

        is_train = rng.random() < 0.8
        if is_train:
            plt.savefig(os.path.join(train_path, data['filename']), dpi=96)
            train_rows.append(data)
        else:
            plt.savefig(os.path.join(test_path, data['filename']), dpi=96)
            test_rows.append(data)
        plt.close(fig)

    pd.DataFrame(train_rows).to_csv(os.path.join(base_path, 'train_metadata.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(base_path, 'test_metadata.csv'), index=False)
    print(f"Finished {dataset_name}: Train {len(train_rows)}, Test {len(test_rows)}")

def generate_flow_dataset(noise_type: str, noise_ratio: float = 0.1, root_dir: str = './data'):
    dataset_name = f'flow_{noise_type}'
    base_path = os.path.join(root_dir, dataset_name)
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print(f"Generating Flow ({noise_type}) to {base_path}...")
    df_det = pd.DataFrame(get_flow_det())
    noise_std_h = df_det['h_deterministic'].std() * noise_ratio
    noise_std_x = df_det['x_true_deterministic'].std() * noise_ratio

    mean_r, std_r = df_det['r'].mean(), df_det['r'].std()
    mean_hole, std_hole = df_det['hole'].mean(), df_det['hole'].std()

    rng = np.random.default_rng(seed=42)
    train_rows, test_rows = [], []

    for idx, det_row in df_det.iterrows():
        r_val = det_row['r']
        hole_val = det_row['hole']
        ball_r = det_row['ball_r']
        hole_pos_y = det_row['hole_pos_y']

        pa_dist = np.sqrt(((r_val - mean_r)/std_r)**2 + ((hole_val - mean_hole)/std_hole)**2) / np.sqrt(2.0)

        if noise_type == 'hetero':
            scale_factor = 0.5 + pa_dist
            eps_h = rng.normal(0.0, 1.0)
            eps_x = rng.normal(0.0, 1.0)
            h_final = max(0.0, det_row['h_deterministic'] + noise_std_h * scale_factor * eps_h)
            x_final = max(0.0, det_row['x_true_deterministic'] + noise_std_x * scale_factor * eps_x)

        elif noise_type == 'mult':
            eps_h = rng.normal(0.0, 1.0)
            eps_x = rng.normal(0.0, 1.0)
            h_final = max(0.0, det_row['h_deterministic'] * np.exp(noise_ratio * eps_h))
            x_final = max(0.0, det_row['x_true_deterministic'] * np.exp(noise_ratio * eps_x))

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        fig, ax, data = gen_flow_img(int(r_val), int(hole_val), h_final, x_final, ball_r, hole_pos_y)
        data['filename'] = f"{idx}.png"

        is_train = rng.random() < 0.8
        if is_train:
            plt.savefig(os.path.join(train_path, data['filename']), dpi=96)
            train_rows.append(data)
        else:
            plt.savefig(os.path.join(test_path, data['filename']), dpi=96)
            test_rows.append(data)
        plt.close(fig)

    pd.DataFrame(train_rows).to_csv(os.path.join(base_path, 'train_metadata.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(base_path, 'test_metadata.csv'), index=False)
    print(f"Finished {dataset_name}: Train {len(train_rows)}, Test {len(test_rows)}")

def main():
    print("Generating Non-ANM Datasets (Heteroscedastic and Multiplicative)...")
    tasks = [
        ('pendulum', 'hetero'),
        ('pendulum', 'mult'),
        ('flow', 'hetero'),
        ('flow', 'mult')
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for ds, n_type in tasks:
            if ds == 'pendulum':
                futures.append(executor.submit(generate_pendulum_dataset, n_type))
            else:
                futures.append(executor.submit(generate_flow_dataset, n_type))
        for f in futures:
            f.result()
    print("All non-ANM datasets generated successfully!")

if __name__ == "__main__":
    main()

