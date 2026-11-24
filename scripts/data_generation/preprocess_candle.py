import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Config ---
# Assumes user unzips the IITH_CANDLE dataset into data/raw/IITH_CANDLE or similar
RAW_DATA_ROOT = "./data/IITH_CANDLE" 
OUTPUT_DIR = "./data/candle" 

mappings = {
    "scene": {
        "indoor": 0, "playground": 1, "outdoor": 2, "bridge": 3, "city square": 4, 
        "hall": 5, "grassland": 6, "garage": 7, "street": 8, "beach": 9, 
        "station": 10, "tunnel": 11, "moonlit grass": 12, "dusk city": 13, 
        "skywalk": 14, "garden": 15
    },
    "lights": {"left": 0, "middle": 1, "right": 2},
    "object_type": {"cube": 0, "sphere": 1, "cylinder": 2, "cone": 3, "torus": 4},
    "color": {"red": 0, "blue": 1, "yellow": 2, "purple": 3, "orange": 4},
    "size": {1.5: 0, 2.0: 1, 2.5: 2}, 
    "rotation": {0: 0, 15: 1, 30: 2, 45: 3, 60: 4, 90: 5}
}

def process_candle():
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"Error: Raw data root {RAW_DATA_ROOT} not found.")
        print("Please unzip IITH_CANDLE into the data/ folder first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_rows = []
    
    files = sorted([f for f in os.listdir(RAW_DATA_ROOT) if f.endswith('.json')])
    print(f"Found {len(files)} JSON files in {RAW_DATA_ROOT}. Processing...")

    for f_name in files:
        base_name = f_name.replace('.json', '')
        img_name = f"{base_name}.png"
        
        # Handle .jpg case just in case
        if not os.path.exists(os.path.join(RAW_DATA_ROOT, img_name)):
             img_name = f"{base_name}.jpg"

        try:
            with open(os.path.join(RAW_DATA_ROOT, f_name), 'r') as f:
                meta = json.load(f)
            
            # Get object data (e.g., "Cube_0")
            obj_key = list(meta['objects'].keys())[0]
            obj_data = meta['objects'][obj_key]

            # --- NEW: Extract Bounding Box ---
            # CANDLE format: [[min_x, min_y], [max_x, max_y]]
            bounds = obj_data['bounds'] 
            min_x, min_y = bounds[0]
            max_x, max_y = bounds[1]
            
            data_rows.append({
                'filename': img_name,
                'light': mappings['lights'][meta['lights']],
                'scene': mappings['scene'][meta['scene']],
                'object': mappings['object_type'][obj_data['object_type']],
                'color': mappings['color'][obj_data['color']],
                'size': mappings['size'][obj_data['size']],
                'rotation': mappings['rotation'][obj_data['rotation']],
                # Save raw coords
                'min_x': min_x, 'min_y': min_y, 
                'max_x': max_x, 'max_y': max_y
            })
        except Exception as e:
            print(f"Skipping {f_name}: {e}")

    df = pd.DataFrame(data_rows)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"), index=False)
    
    print(f"Metadata saved to {OUTPUT_DIR}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    process_candle()