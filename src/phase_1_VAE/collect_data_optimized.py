import pandas as pd
import numpy as np
import pathlib
import sys
import tqdm

# Add folders to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent.parent / "Hackathon_student"))

from dataset import run_single_simulation, build_histogram, OPTIMIZE

def collect_from_csv_optimized():
    student_dir = CURRENT_DIR.parent.parent / "Hackathon_student"
    csv_path = student_dir / "beamline_results.csv"
    output_dataset_path = student_dir / "beamline_dataset.npz"
    
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return
        
    print(f"Reading trials from {csv_path}...")
    df = pd.read_csv(csv_path)
    df_completed = df[df['state'] == 'COMPLETE']
    print(f"Found {len(df_completed)} completed trials in CSV.")
    
    # Load existing dataset if it exists
    old_voltages = np.empty((0, 8), dtype=np.float32)
    old_histograms = np.empty((0, 400), dtype=np.float32)
    if output_dataset_path.exists():
        try:
            old_data = np.load(output_dataset_path)
            old_voltages = old_data['voltages'].astype(np.float32)
            old_histograms = old_data['histograms'].astype(np.float32)
            print(f"Loaded existing dataset: voltages {old_voltages.shape}, histograms {old_histograms.shape}")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
            
    voltages_list = []
    histograms_list = []
    
    opt_keys = sorted(list(OPTIMIZE.keys()))
    print(f"Optimized electrodes order: {['V' + str(k) for k in opt_keys]}")
    
    new_simulations_count = 0
    
    for idx, row in tqdm.tqdm(df_completed.iterrows(), total=len(df_completed), desc="Processing trials"):
        # Reconstruct chosen voltages dict
        chosen = {}
        for k in opt_keys:
            col_name = f"params_V{k}"
            if col_name in row:
                chosen[k] = float(row[col_name])
            else:
                chosen[k] = 0.0 # fallback
                
        voltages_vector = np.array([chosen[k] for k in opt_keys], dtype=np.float32)
        
        # Check if this voltage vector already exists in our loaded old_voltages
        found_match = False
        if len(old_voltages) > 0:
            # Check proximity to handle potential small float rounding diffs
            diffs = np.linalg.norm(old_voltages - voltages_vector, axis=1)
            min_diff_idx = np.argmin(diffs)
            if diffs[min_diff_idx] < 1e-3:
                # Use the existing histogram
                hist = old_histograms[min_diff_idx]
                found_match = True
                
        if found_match:
            voltages_list.append(voltages_vector)
            histograms_list.append(hist)
        else:
            # We need to run SIMION for this new trial
            try:
                positions = run_single_simulation(chosen)
                hist = build_histogram(positions)
                voltages_list.append(voltages_vector)
                histograms_list.append(hist)
                new_simulations_count += 1
            except Exception as e:
                print(f"\nError simulating new trial {row.get('number', idx)}: {e}")
                continue
                
    if len(voltages_list) == 0:
        print("No data collected.")
        return
        
    voltages_array = np.stack(voltages_list)       # (N, 8)
    histograms_array = np.stack(histograms_list)   # (N, 400)
    
    # Save as .npz file
    np.savez_compressed(
        output_dataset_path, 
        voltages=voltages_array, 
        histograms=histograms_array
    )
    print(f"Dataset updated. Total samples: {voltages_array.shape[0]} (Added {new_simulations_count} new simulations).")
    print(f"Saved to {output_dataset_path}")

if __name__ == "__main__":
    collect_from_csv_optimized()
