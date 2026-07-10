import os
import sys
import pathlib
import numpy as np
import pandas as pd
import tqdm

# Setup paths
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent.parent / "Hackathon_student"))

from dataset import run_single_simulation, build_histogram, OPTIMIZE

def collect_all_real_data():
    student_dir = CURRENT_DIR.parent.parent / "Hackathon_student"
    dataset_path = student_dir / "beamline_dataset.npz"
    perturbed_csv_path = student_dir / "voltajes_perturbados_lhs.csv"
    
    print("=========================================================")
    print("  SIMION DATA COLLECTION: LHS + LOCAL MICRO-PERTURBATIONS")
    print("=========================================================")
    
    # 1. Load existing data if available
    old_voltages = np.empty((0, 8), dtype=np.float32)
    old_histograms = np.empty((0, 400), dtype=np.float32)
    if dataset_path.exists():
        try:
            data = np.load(dataset_path)
            old_voltages = data['voltages'].astype(np.float32)
            old_histograms = data['histograms'].astype(np.float32)
            print(f"Loaded existing dataset: {old_voltages.shape[0]} samples.")
        except Exception as e:
            print(f"Warning: Could not load existing dataset ({e}).")
            
    # 2. Load local perturbations (Módulo 4 output)
    perturbed_voltages = np.empty((0, 8), dtype=np.float32)
    if perturbed_csv_path.exists():
        try:
            df_pert = pd.read_csv(perturbed_csv_path)
            voltage_cols = [c for c in df_pert.columns if c.startswith('V')]
            perturbed_voltages = df_pert[voltage_cols].values.astype(np.float32)
            print(f"Loaded {perturbed_voltages.shape[0]} local perturbed voltage samples.")
        except Exception as e:
            print(f"Error loading perturbed CSV: {e}")
    else:
        print("Warning: Local perturbed CSV not found. Make sure to generate it first.")
        
    # 3. Generate LHS samples (Módulo 1)
    from scipy.stats.qmc import LatinHypercube
    n_lhs = 330
    bounds_min = np.array([-1000.0] * 8, dtype=np.float32)
    bounds_max = np.array([1000.0] * 8, dtype=np.float32)
    
    sampler = LatinHypercube(d=8, seed=42)
    lhs_raw = sampler.random(n=n_lhs)
    lhs_voltages = (bounds_min + lhs_raw * (bounds_max - bounds_min)).astype(np.float32)
    print(f"Generated {n_lhs} Latin Hypercube Sampling (LHS) voltage configurations.")
    
    # 4. Merge target voltage configurations to simulate
    # We combine LHS samples and perturbed samples
    candidate_voltages = np.vstack([lhs_voltages, perturbed_voltages])
    print(f"Total candidate configurations to evaluate: {candidate_voltages.shape[0]}")
    
    opt_keys = sorted(list(OPTIMIZE.keys()))
    
    voltages_list = []
    histograms_list = []
    
    # We also keep existing data to avoid re-simulating
    if len(old_voltages) > 0:
        voltages_list = list(old_voltages)
        histograms_list = list(old_histograms)
        
    new_simulations_count = 0
    cached_count = 0
    
    # Simulate sequentially to avoid file conflicts in fastadj
    for i in tqdm.tqdm(range(len(candidate_voltages)), desc="Evaluating configurations"):
        v_vector = candidate_voltages[i]
        
        # Check if we already have this voltage configuration in our accumulated list
        found_match = False
        if len(voltages_list) > 0:
            diffs = np.linalg.norm(np.array(voltages_list) - v_vector, axis=1)
            min_diff_idx = np.argmin(diffs)
            if diffs[min_diff_idx] < 1e-2:
                # Cache hit
                cached_count += 1
                found_match = True
                # If it's already in the accumulated list, we don't need to append it again 
                # unless we want duplicates (we don't want duplicates).
                
        if found_match:
            continue
            
        # If not found, run SIMION
        chosen = {opt_keys[j]: float(v_vector[j]) for j in range(8)}
        try:
            positions = run_single_simulation(chosen)
            hist = build_histogram(positions)
            voltages_list.append(v_vector)
            histograms_list.append(hist)
            new_simulations_count += 1
        except Exception as e:
            print(f"\nError simulating configuration {i}: {e}")
            continue
            
    # Save the updated compressed dataset
    updated_voltages = np.stack(voltages_list)
    updated_histograms = np.stack(histograms_list)
    
    np.savez_compressed(
        dataset_path,
        voltages=updated_voltages,
        histograms=updated_histograms
    )
    
    print("\n=========================================================")
    print("  DATA COLLECTION COMPLETED SUCCESSFULLY!")
    print("=========================================================")
    print(f"Total samples in dataset: {updated_voltages.shape[0]}")
    print(f"New SIMION simulations performed: {new_simulations_count}")
    print(f"Pre-existing / Cached samples: {cached_count}")
    print(f"Dataset saved to: {dataset_path}")
    print("=========================================================")

if __name__ == "__main__":
    collect_all_real_data()
