import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

# --- Constants ---
BALL_R_RANGE = range(5, 35)
H_RAW_RANGE = range(10, 40)
HOLE_POS_RANGE = range(6, 15)
GRAVITY_CONST = 0.98
BALL_R_SCALE = 30.0
H_RAW_SCALE = 10.0
HOLE_POS_SCALE = 3.0
CUP_X_LEFT, CUP_X_RIGHT = 3.0, 8.0
CUP_WIDTH = CUP_X_RIGHT - CUP_X_LEFT
CUP_TOP_Y, CUP_BOTTOM_Y = 19.0, 0.0
BALL_CENTER_X = 5.5
BALL_Y_OFFSET = 0.5
HOLE_HEIGHT = 0.4
WATER_LINE_Y_MIN = 0.5
FIGURE_X_LIM, FIGURE_Y_LIM = (0, 20), (0, 20)

def get_deterministic_metadata() -> List[Dict[str, Any]]:
    metadata_rows = []
    for r in BALL_R_RANGE:
        for h_raw in H_RAW_RANGE:
            for hole in HOLE_POS_RANGE:
                ball_r = r / BALL_R_SCALE
                h_det = pow(ball_r, 3) + (h_raw / H_RAW_SCALE)
                hole_pos_y = hole / HOLE_POS_SCALE
                h_eff = max(0, h_det - (hole_pos_y - WATER_LINE_Y_MIN))
                x_true_det = np.sqrt(2 * GRAVITY_CONST * h_eff * (hole_pos_y - WATER_LINE_Y_MIN))
                metadata_rows.append({
                    'r': r, 'h_raw': h_raw, 'hole': hole,
                    'h_deterministic': h_det,
                    'x_true_deterministic': x_true_det,
                    'hole_pos_y': hole_pos_y, 'ball_r': ball_r
                })
    return metadata_rows

def generate_single_image(ball_r_i, hole_pos_i, h_final, x_true_final, ball_r, hole_pos_y):
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.add_artist(plt.Rectangle((CUP_X_LEFT, CUP_BOTTOM_Y), CUP_WIDTH, 5 + h_final, color='lightskyblue'))
    ax.add_artist(plt.Circle((BALL_CENTER_X, ball_r + BALL_Y_OFFSET), ball_r, color='firebrick'))
    ax.add_artist(plt.Polygon(([CUP_X_LEFT, CUP_BOTTOM_Y], [CUP_X_LEFT, CUP_TOP_Y]), color='black', linewidth=2))
    ax.add_artist(plt.Polygon(([CUP_X_RIGHT, CUP_BOTTOM_Y], [CUP_X_RIGHT, hole_pos_y]), color='black', linewidth=2))
    ax.add_artist(plt.Polygon(([CUP_X_RIGHT, hole_pos_y + HOLE_HEIGHT], [CUP_X_RIGHT, CUP_TOP_Y]), color='black', linewidth=2))
    
    y_flow = np.linspace(hole_pos_y, WATER_LINE_Y_MIN)
    h_eff_plot = max(0, h_final - (hole_pos_y - WATER_LINE_Y_MIN))
    x_flow = np.sqrt(2 * GRAVITY_CONST * h_eff_plot * (hole_pos_y - y_flow)) + CUP_X_RIGHT
    if x_flow.max() > CUP_X_RIGHT:
        scale_factor = (x_true_final / (x_flow.max() - CUP_X_RIGHT + 1e-8))
        x_flow = (x_flow - CUP_X_RIGHT) * scale_factor + CUP_X_RIGHT
    
    plt.plot(x_flow, y_flow, color='lightskyblue', linewidth=2)
    plt.plot(np.linspace(0, 20, 50), np.zeros(50) + 0.2, color='black', linewidth=2)
    ax.set_xlim(FIGURE_X_LIM)
    ax.set_ylim(FIGURE_Y_LIM)
    ax.axis('off')
    return fig, ax, {'r': ball_r_i, 'h': h_final, 'x_true_10x': x_true_final * 10, 'hole': hole_pos_i}

def generate_dataset(root_dir='./data', dataset_suffix='', noise_ratio=0.0, save_fig=True):
    dataset_name = f'flow{dataset_suffix}'
    base_path = os.path.join(root_dir, dataset_name)
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    if save_fig:
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

    print(f"Generating {dataset_name} to {base_path} (noise_ratio={noise_ratio} | UNIFORM noise)...")
    df_det = pd.DataFrame(get_deterministic_metadata())
    
    # Calculate target standard deviation
    noise_std_h = df_det['h_deterministic'].std() * noise_ratio
    noise_std_x = df_det['x_true_deterministic'].std() * noise_ratio
    
    # --- UNIFORM NOISE CALCULATION ---
    # Set bound 'a' such that std_dev matches Gaussian: a = std * sqrt(3)
    h_bound = noise_std_h * np.sqrt(3)
    x_bound = noise_std_x * np.sqrt(3)
    
    rng = np.random.default_rng(seed=42)
    train_rows, test_rows = [], []

    for idx, det_row in df_det.iterrows():
        # Using Uniform noise instead of Gaussian
        if noise_ratio > 0:
            h_noise = rng.uniform(-h_bound, h_bound)
            x_noise = rng.uniform(-x_bound, x_bound)
        else:
            h_noise, x_noise = 0.0, 0.0

        h_final = max(0, det_row['h_deterministic'] + h_noise)
        x_final = max(0, det_row['x_true_deterministic'] + x_noise)
        
        fig, ax, data = generate_single_image(int(det_row['r']), int(det_row['hole']), h_final, x_final, det_row['ball_r'], det_row['hole_pos_y'])
        data['filename'] = f"{idx}.png"
        
        if save_fig:
            # --- FIX: Sample ONCE ---
            if rng.random() < 0.8:
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
    # Changed suffix to _uniform to distinguish from Gaussian datasets
    generate_dataset(dataset_suffix='_anm_uniform', noise_ratio=0.1)