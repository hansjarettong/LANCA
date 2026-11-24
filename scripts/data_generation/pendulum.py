import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

# --- Constants ---
PENDULUM_ANGLE_RANGE = range(-40, 44)
LIGHT_ANGLE_RANGE = range(60, 148)
LIGHT_ANGLE_SKIP = 100

PENDULUM_PIVOT = (10.0, 10.5)
PENDULUM_LENGTH = 8.0
PENDULUM_BALL_RADIUS = 1.5
PENDULUM_ROD_LENGTH = 9.5
LIGHT_SOURCE_Y = 20.5
LIGHT_SOURCE_RADIUS = 3.0
SHADOW_Y_POSITION = -0.5
MIN_SHADOW_LENGTH = 3.0
FIGURE_X_LIM = (0, 20)
FIGURE_Y_LIM = (-1, 21)

NORMALIZATION_SCALE = {
    'i': (0.0, 44.0),
    'j': (100.0, 40.0),
    'shade': (7.0, 7.5),
    'mid': (7.0, 7.5)
}

def projection(phi: float, x: float, y: float, base: float = -0.5) -> float:
    b = y - x * math.tan(phi)
    shade_x = (base - b) / math.tan(phi)
    return shade_x

def get_deterministic_metadata() -> List[Dict[str, Any]]:
    metadata_rows = []
    for i in PENDULUM_ANGLE_RANGE:
        for j in LIGHT_ANGLE_RANGE:
            if j == LIGHT_ANGLE_SKIP:
                continue
                
            theta = i * math.pi / 200.0
            phi = j * math.pi / 200.0
            
            proj_pivot = projection(phi, PENDULUM_PIVOT[0], PENDULUM_PIVOT[1], SHADOW_Y_POSITION)
            
            rod_end_x = PENDULUM_PIVOT[0] + PENDULUM_ROD_LENGTH * math.sin(theta)
            rod_end_y = PENDULUM_PIVOT[1] - PENDULUM_ROD_LENGTH * math.cos(theta)
            proj_rod_end = projection(phi, rod_end_x, rod_end_y, SHADOW_Y_POSITION)
            
            shade_deterministic = max(MIN_SHADOW_LENGTH, abs(proj_pivot - proj_rod_end))
            mid_deterministic = (proj_pivot + proj_rod_end) / 2.0

            metadata_rows.append({
                'i': i, 'j': j,
                'shade_deterministic': shade_deterministic,
                'mid_deterministic': mid_deterministic
            })
    return metadata_rows

def generate_single_image(i, j, shade_final, mid_final) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    theta = i * math.pi / 200.0
    phi = j * math.pi / 200.0
    
    x = PENDULUM_PIVOT[0] + PENDULUM_LENGTH * math.sin(theta)
    y = PENDULUM_PIVOT[1] - PENDULUM_LENGTH * math.cos(theta)
    light_x = projection(phi, PENDULUM_PIVOT[0], PENDULUM_PIVOT[1], LIGHT_SOURCE_Y)
    
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    
    ball = plt.Circle((x, y), PENDULUM_BALL_RADIUS, color='firebrick')
    gun = plt.Polygon((PENDULUM_PIVOT, [x, y]), color='black', linewidth=3)
    sun = plt.Circle((light_x, LIGHT_SOURCE_Y), LIGHT_SOURCE_RADIUS, color='orange')
    shadow_poly = plt.Polygon(
        ([mid_final - shade_final / 2.0, SHADOW_Y_POSITION], [mid_final + shade_final / 2.0, SHADOW_Y_POSITION]),
        color='black', linewidth=3
    )
    
    ax.add_artist(gun)
    ax.add_artist(ball)
    ax.add_artist(sun)
    ax.add_artist(shadow_poly)
    ax.set_xlim(FIGURE_X_LIM)
    ax.set_ylim(FIGURE_Y_LIM)
    ax.axis('off')
    
    metadata = {'i': i, 'j': j, 'shade': shade_final, 'mid': mid_final}
    return fig, ax, metadata

def generate_dataset(root_dir='./data', dataset_suffix='', noise_ratio=0.0, save_fig=True):
    dataset_name = f'pendulum{dataset_suffix}'
    base_path = os.path.join(root_dir, dataset_name)
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    if save_fig:
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
    print(f"Generating {dataset_name} to {base_path} (noise_ratio={noise_ratio})...")

    df_det = pd.DataFrame(get_deterministic_metadata())
    noise_std_shade = df_det['shade_deterministic'].std() * noise_ratio
    noise_std_mid = df_det['mid_deterministic'].std() * noise_ratio
    
    rng = np.random.default_rng(seed=42)
    train_rows, test_rows = [], []
    
    for idx, det_row in df_det.iterrows():
        shade_noise = rng.normal(0.0, noise_std_shade) if noise_ratio > 0 else 0.0
        mid_noise = rng.normal(0.0, noise_std_mid) if noise_ratio > 0 else 0.0
        
        shade_final = max(MIN_SHADOW_LENGTH, det_row['shade_deterministic'] + shade_noise)
        mid_final = det_row['mid_deterministic'] + mid_noise
        
        fig, ax, data = generate_single_image(
            int(det_row['i']), int(det_row['j']), shade_final, mid_final
        )
        
        data['filename'] = f"{idx}.png"
        for k, (min_v, range_v) in NORMALIZATION_SCALE.items():
             data[f'{k}_norm'] = (data[k] - min_v) / range_v

        if save_fig:
            # --- FIX: Sample ONCE ---
            is_train = rng.random() < 0.8
            
            if is_train:
                plt.savefig(os.path.join(train_path, data['filename']), dpi=96)
                train_rows.append(data)
            else:
                plt.savefig(os.path.join(test_path, data['filename']), dpi=96)
                test_rows.append(data)
        
        plt.close(fig)
        if idx % 1000 == 0: print(f"Generated {idx} images...")

    pd.DataFrame(train_rows).to_csv(os.path.join(base_path, 'train_metadata.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(base_path, 'test_metadata.csv'), index=False)
    print(f"Done. Train: {len(train_rows)}, Test: {len(test_rows)}")

if __name__ == "__main__":
    generate_dataset(dataset_suffix='_anm_nr10', noise_ratio=0.1)