import pandas as pd
import numpy as np
import pathlib
import sys
import tqdm

# Add src/phase_1_VAE and Hackathon_student to path
# __file__ is src/phase_1_VAE/collect_data.py -> CURRENT_DIR is src/phase_1_VAE -> CURRENT_DIR.parent.parent is workspace root
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent.parent / "Hackathon_student"))

from dataset import run_single_simulation, build_histogram, OPTIMIZE

def collect_from_csv():
    student_dir = CURRENT_DIR.parent.parent / "Hackathon_student"
    csv_path = student_dir / "beamline_results.csv"
    output_dataset_path = student_dir / "beamline_dataset.npz"
    
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return
        
    print(f"Reading trials from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter completed trials
    df_completed = df[df['state'] == 'COMPLETE']
    print(f"Found {len(df_completed)} completed trials to process.")
    
    voltages_list = []
    histograms_list = []
    
    # Sort optimization keys to ensure consistent order:
    # V3, V6, V9, V10, V11, V12, V15, V18
    opt_keys = sorted(list(OPTIMIZE.keys()))
    print(f"Optimized electrodes order: {['V' + str(k) for k in opt_keys]}")
    
    for idx, row in tqdm.tqdm(df_completed.iterrows(), total=len(df_completed), desc="Simulating trials"):
        # Reconstruct chosen voltages dict
        chosen = {}
        for k in opt_keys:
            col_name = f"params_V{k}"
            if col_name in row:
                chosen[k] = float(row[col_name])
            else:
                chosen[k] = 0.0 # fallback
                
        # Run SIMION simulation
        try:
            positions = run_single_simulation(chosen)
            # Build 2D histogram
            hist = build_histogram(positions)
            
            # Save to lists
            voltages_vector = np.array([chosen[k] for k in opt_keys], dtype=np.float32)
            voltages_list.append(voltages_vector)
            histograms_list.append(hist)
        except Exception as e:
            print(f"\nError simulating trial {row.get('number', idx)}: {e}")
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
    print(f"Saved dataset of shape voltages: {voltages_array.shape}, histograms: {histograms_array.shape} to {output_dataset_path}")

if __name__ == "__main__":
    collect_from_csv()
