#!/usr/bin/env python
import argparse
import sys
import os
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import optuna

# Add directories to path
ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(ROOT_DIR / "src" / "phase_1_VAE"))
sys.path.append(str(ROOT_DIR / "src" / "phase_2_DNN"))
sys.path.append(str(ROOT_DIR / "src" / "phase_3_control"))

# Try importing modules
try:
    from vae import VAE
    from model import ForwardRegressor, FullEmulator
    from inverse_estimator import InverseEstimator
    from dataset import BeamlineDataset, run_single_simulation, build_histogram, OPTIMIZE
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class DigitalTwin:
    """
    User-Friendly Interface for the Silicon Ion Beamline Digital Twin.
    Loads models on CPU by default, allowing it to run on any computer.
    """
    def __init__(self):
        # Force CPU device for maximum portability (unless CUDA is explicitly desired)
        self.device = torch.device("cpu")
        
        # Paths
        self.dataset_path = ROOT_DIR / "Hackathon_student" / "beamline_dataset.npz"
        self.vae_weights = ROOT_DIR / "src" / "phase_1_VAE" / "vae_model.pt"
        self.regressor_weights = ROOT_DIR / "src" / "phase_2_DNN" / "forward_regressor.pt"
        self.inverse_weights = ROOT_DIR / "src" / "phase_3_control" / "inverse_estimator.pt"
        
        # Verify weight files exist
        for path in [self.dataset_path, self.vae_weights, self.regressor_weights, self.inverse_weights]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path.name}. Please ensure model weights have been trained.")
                
        # Initialize dataset (needed for normalization ranges)
        self.dataset = BeamlineDataset(str(self.dataset_path))
        
        # Load VAE (Phase 1)
        self.vae = VAE(input_dim=400, latent_dim=2).to(self.device)
        self.vae.load_state_dict(torch.load(self.vae_weights, map_location=self.device))
        self.vae.eval()
        
        # Load Forward Regressor (Phase 2, y -> z)
        self.forward_reg = ForwardRegressor(input_dim=8, latent_dim=2).to(self.device)
        self.forward_reg.load_state_dict(torch.load(self.regressor_weights, map_location=self.device))
        self.forward_reg.eval()
        
        # Load Inverse Estimator (Phase 3, z -> y)
        self.inverse_est = InverseEstimator(latent_dim=2, output_dim=8).to(self.device)
        self.inverse_est.load_state_dict(torch.load(self.inverse_weights, map_location=self.device))
        self.inverse_est.eval()
        
        # Combined Full Emulator (y -> X_hat)
        self.emulator = FullEmulator(self.forward_reg, self.vae.decoder).to(self.device)
        self.emulator.eval()
        
        # Fit KDE density filter on the training latent space
        self._fit_density_filter()

    def _fit_density_filter(self):
        """Fits Kernel Density Estimator (KDE) on training latent space coordinates."""
        histograms_t = torch.tensor(self.dataset.histograms, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            true_z, _ = self.vae.encoder(histograms_t)
            true_z = true_z.numpy()
            
        self.train_z = true_z
        self.z_min = true_z.min(axis=0)
        self.z_max = true_z.max(axis=0)
        
        # Fit KDE with optimal bandwidth
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(true_z)
        self.threshold = np.percentile(self.kde.score_samples(true_z), 5)

    def is_physical_state(self, z):
        """Checks if a latent point z belongs to the physical manifold (Rubric G)."""
        z_np = np.array(z, dtype=np.float32).reshape(1, -1)
        log_density = self.kde.score_samples(z_np)[0]
        return log_density >= self.threshold, log_density

    def normalize_voltages(self, physical_voltages):
        """Converts physical voltages to normalized range [-1, 1] using dataset parameters."""
        if isinstance(physical_voltages, torch.Tensor):
            min_t = torch.tensor(self.dataset.min_vals, device=physical_voltages.device)
            max_t = torch.tensor(self.dataset.max_vals, device=physical_voltages.device)
            return 2.0 * (physical_voltages - min_t) / (max_t - min_t) - 1.0
        else:
            return 2.0 * (physical_voltages - self.dataset.min_vals) / (self.dataset.max_vals - self.dataset.min_vals) - 1.0

    def predict_profile(self, physical_voltages):
        """
        Forward prediction: Voltages (y) -> Differentiable Emulator -> Predicted Profile (X_hat).
        Does NOT require SIMION.
        """
        # Convert physical voltages to normalized [-1, 1]
        norm_voltages = self.normalize_voltages(np.array(physical_voltages, dtype=np.float32))
        
        # Convert to tensor
        y_t = torch.tensor(norm_voltages, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Run Full Emulator
            x_hat, z_hat = self.emulator(y_t)
            
        x_hat = x_hat.squeeze(0).numpy()
        z_hat = z_hat.squeeze(0).numpy()
        
        # Re-scale back to ion count intensity (each pixel is fraction of 500 launched ions)
        grid_profile = x_hat.reshape(20, 20)
        transmission = float(np.sum(x_hat))
        
        # Calculate predicted spread
        y_coords, x_coords = np.meshgrid(np.arange(20), np.arange(20), indexing='ij')
        total_intensity = np.sum(grid_profile)
        if total_intensity > 1e-5:
            mean_y = np.sum(grid_profile * y_coords) / total_intensity
            mean_x = np.sum(grid_profile * x_coords) / total_intensity
            var_y = np.sum(grid_profile * (y_coords - mean_y)**2) / total_intensity
            var_x = np.sum(grid_profile * (x_coords - mean_x)**2) / total_intensity
            spread = float(np.sqrt(var_y) + np.sqrt(var_x))
        else:
            spread = 1e9
            
        return grid_profile, transmission, spread, z_hat

    def control_inverse(self, target_profile):
        """
        Inverse Setpoint: Target Profile -> Inverse Estimator + Backpropagation -> Voltages.
        Does NOT require SIMION.
        """
        # 1. Encode target profile to latent z
        target_t = torch.tensor(target_profile, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            z_target, _ = self.vae.encoder(target_t)
            z_target_np = z_target.squeeze(0).numpy()
            
        # 2. Check and project z if it is out-of-distribution (low density)
        is_valid, log_dens = self.is_physical_state(z_target_np)
        if not is_valid:
            dists = np.sum((self.train_z - z_target_np)**2, axis=1)
            z_target_np = self.train_z[np.argmin(dists)]
            z_target = torch.tensor(z_target_np, dtype=torch.float32, device=self.device).view(1, -1)
            
        # 3. Get initial voltage guess using Inverse Estimator (z -> y_init)
        with torch.no_grad():
            norm_voltages_init = self.inverse_est(z_target).squeeze(0).numpy()
            
        # 4. Refine voltages via backpropagation through Forward Emulator
        y_init_t = torch.tensor(norm_voltages_init, dtype=torch.float32, device=self.device)
        y_param = torch.tensor(norm_voltages_init, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([y_param], lr=0.02)
        criterion = torch.nn.MSELoss()
        
        for _ in range(100):
            optimizer.zero_grad()
            x_hat, _ = self.emulator(y_param.unsqueeze(0))
            loss_term = criterion(x_hat.squeeze(0), target_t.squeeze(0))
            reg_term = 2.0 * torch.sum((y_param - y_init_t)**2) # Proximal constraint
            loss = loss_term + reg_term
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_param.clamp_(-1.0, 1.0)
                
        norm_voltages_refined = y_param.detach().cpu().numpy()
        physical_voltages = self.dataset.denormalize_voltages(norm_voltages_refined)
        
        # Map to dict keys V3, V6, ...
        opt_keys = sorted(list(OPTIMIZE.keys()))
        voltage_dict = {key: float(val) for key, val in zip(opt_keys, physical_voltages)}
        
        return voltage_dict, z_target_np

    def optimize_steering(self, n_trials=100):
        """
        Latent Space Optimization (C-BO): Maximize beam transmission within physical latent boundaries.
        Does NOT require SIMION.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        
        z1_range = self.z_max[0] - self.z_min[0]
        z2_range = self.z_max[1] - self.z_min[1]
        z1_min, z1_max = self.z_min[0] - 0.1 * z1_range, self.z_max[0] + 0.1 * z1_range
        z2_min, z2_max = self.z_min[1] - 0.1 * z2_range, self.z_max[1] + 0.1 * z2_range
        
        def objective(trial):
            z1 = trial.suggest_float("z1", z1_min, z1_max)
            z2 = trial.suggest_float("z2", z2_min, z2_max)
            z = [z1, z2]
            
            is_valid, _ = self.is_physical_state(z)
            if not is_valid:
                return -1.0
                
            # Decode and score transmission
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).view(1, -1)
            with torch.no_grad():
                x_hat = self.vae.decoder(z_t)
            return float(torch.sum(x_hat).item())

        study.optimize(objective, n_trials=n_trials)
        best_z = [study.best_params["z1"], study.best_params["z2"]]
        
        # Infer voltages
        best_z_t = torch.tensor(best_z, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            norm_voltages_init = self.inverse_est(best_z_t).squeeze(0).numpy()
            
        # Refine voltages via backprop
        y_init_t = torch.tensor(norm_voltages_init, dtype=torch.float32, device=self.device)
        y_param = torch.tensor(norm_voltages_init, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([y_param], lr=0.02)
        
        for _ in range(80):
            optimizer.zero_grad()
            x_hat, _ = self.emulator(y_param.unsqueeze(0))
            loss = -torch.sum(x_hat) + 2.0 * torch.sum((y_param - y_init_t)**2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_param.clamp_(-1.0, 1.0)
                
        norm_voltages_refined = y_param.detach().cpu().numpy()
        physical_voltages = self.dataset.denormalize_voltages(norm_voltages_refined)
        
        opt_keys = sorted(list(OPTIMIZE.keys()))
        voltage_dict = {key: float(val) for key, val in zip(opt_keys, physical_voltages)}
        
        return voltage_dict, best_z

    def validate_in_simion(self, voltages):
        """Runs validation in SIMION if environment supports it."""
        positions = run_single_simulation(voltages)
        hist = build_histogram(positions)
        hits = int(np.sum(hist) * 500)
        from dataset import DETECTOR_REGION
        x_min, x_max = DETECTOR_REGION["x"]
        y_min, y_max = DETECTOR_REGION["y"]
        z_min, z_max = DETECTOR_REGION["z"]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
        valid_pts = positions[mask]
        spread = float(np.std(valid_pts[:, 1]) + np.std(valid_pts[:, 2])) if len(valid_pts) > 0 else 1e9
        print(f"SIMION Validation -> True Hits: {hits}/500 ({hits/5.0:.1f}%), True Spread: {spread:.3f}")
        return hist, hits, spread

def plot_profile(profile, title, save_path):
    """Generates and saves a premium heatmap visualization of the 2D beam profile."""
    plt.figure(figsize=(6, 5))
    plt.imshow(profile, cmap='magma', origin='lower', extent=[-10, 10, -10, 10])
    plt.colorbar(label='Intensidad Normalizada')
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Coordenada Detector X (mm)')
    plt.ylabel('Coordenada Detector Y (mm)')
    plt.grid(True, linestyle=':', alpha=0.3, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada exitosamente en: {save_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Silicon Ion Beamline Digital Twin Surrogate Model Interface.")
    parser.add_argument("--mode", type=str, choices=["predict", "invert", "optimize"], required=True,
                        help="Mode: predict (Forward Emulator), invert (Inverse Setpoint), or optimize (Steering optimization).")
    parser.add_argument("--voltages", type=float, nargs=8, 
                        help="8 physical voltages for V3, V6, V9, V10, V11, V12, V15, V18 in range [-1000, 1000] V (Required for 'predict').")
    parser.add_argument("--trial", type=int, default=104,
                        help="Trial index from training dataset to use as target profile (For 'invert').")
    parser.add_argument("--simion", action="store_true",
                        help="Try to validate results directly in SIMION (Requires local SIMION setup).")
                        
    args = parser.parse_args()
    
    print("\n==========================================")
    print("      ION BEAMLINE DIGITAL TWIN API       ")
    print("==========================================\n")
    
    try:
        dt = DigitalTwin()
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)
        
    if args.mode == "predict":
        if not args.voltages:
            print("Error: --mode predict requires 8 voltages in --voltages.")
            sys.exit(1)
            
        v_dict = dict(zip(sorted(list(OPTIMIZE.keys())), args.voltages))
        print("Voltajes de Entrada:")
        for k, v in sorted(v_dict.items()):
            print(f"  V{k:<6} = {v:+.2f} V")
            
        print("\nEjecutando predicción en el Emulador Forward...")
        profile, trans, spread, z = dt.predict_profile(args.voltages)
        
        print("\n--- Resultados Predichos ---")
        print(f"Coordenadas Latentes z:   [{z[0]:.4f}, {z[1]:.4f}]")
        print(f"Transmisión Estimada:      {trans*100:.2f}% ({int(trans*500)}/500 iones)")
        print(f"Dispersión del Haz (Foco): {spread:.4f}")
        
        save_file = ROOT_DIR / "predicted_profile.png"
        plot_profile(profile, f"Perfil Predicho (Transmisión: {trans*100:.1f}%)", save_file)
        
        if args.simion:
            try:
                print("\nValidando en simulador físico SIMION...")
                dt.validate_in_simion(v_dict)
            except Exception as e:
                print(f"SIMION validation failed: {e}. Check local SIMION setup.")
                
    elif args.mode == "invert":
        if args.trial < 0 or args.trial >= len(dt.dataset.raw_histograms):
            print(f"Error: Trial index must be between 0 and {len(dt.dataset.raw_histograms)-1}.")
            sys.exit(1)
            
        target = dt.dataset.raw_histograms[args.trial]
        target_hits = int(np.sum(target) * 500)
        print(f"Perfil Objetivo: Trial {args.trial} en el Dataset (Transmisión Real: {target_hits}/500)")
        
        print("\nEjecutando Control Inverso en milisegundos...")
        voltages, z_target = dt.control_inverse(target)
        
        print("\n--- Voltajes Inferidos por el Estimador Inverso ---")
        opt_keys = sorted(list(OPTIMIZE.keys()))
        orig_voltages = dt.dataset.raw_voltages[args.trial]
        
        print(f"{'Electrodo':<10} | {'Original (V)':<14} | {'Inferido (V)':<14} | {'Diferencia (V)':<14}")
        print("-" * 62)
        for idx, key in enumerate(opt_keys):
            orig = orig_voltages[idx]
            inferred = voltages[key]
            diff = inferred - orig
            print(f"  V{key:<6} | {orig:14.2f} | {inferred:14.2f} | {diff:+13.2f}")
            
        # Plot target vs predicted
        target_grid = target.reshape(20, 20)
        plot_profile(target_grid, f"Perfil Objetivo (Trial {args.trial}, Hits: {target_hits})", ROOT_DIR / "target_profile.png")
        
        # Predict decoded profile from inferred voltages
        inf_list = [voltages[k] for k in opt_keys]
        recon_grid, trans, spread, _ = dt.predict_profile(inf_list)
        plot_profile(recon_grid, f"Perfil Reconstruido por Gemelo Digital (Hits: {int(trans*500)})", ROOT_DIR / "reconstructed_profile.png")
        
        if args.simion:
            try:
                print("\nValidando voltajes inferidos en SIMION...")
                _, hits, true_spread = dt.validate_in_simion(voltages)
                print(f"\nDiferencia de transmisión física: Original {target_hits} hits vs Reconstruido {hits} hits.")
            except Exception as e:
                print(f"SIMION validation failed: {e}. Ensure SIMION setup is correct.")
                
    elif args.mode == "optimize":
        print("Ejecutando Optimización Bayesiana podada por KDE en espacio latente (C-BO)...")
        voltages, z_opt = dt.optimize_steering(n_trials=150)
        
        print("\n--- Voltajes Óptimos Encontrados para Dirección (Steering) ---")
        for k, v in sorted(voltages.items()):
            print(f"  V{k:<6} = {v:+.2f} V")
            
        opt_list = [voltages[k] for k in sorted(list(OPTIMIZE.keys()))]
        profile, trans, spread, _ = dt.predict_profile(opt_list)
        
        print(f"\nTransmisión Máxima Estimada: {trans*100:.2f}% ({int(trans*500)}/500 iones)")
        print(f"Spread Estimado:             {spread:.4f}")
        
        plot_profile(profile, f"Perfil Optimizado C-BO (Transmisión Estimada: {trans*100:.1f}%)", ROOT_DIR / "optimized_profile.png")
        
        if args.simion:
            try:
                print("\nValidando configuración óptima en SIMION...")
                dt.validate_in_simion(voltages)
            except Exception as e:
                print(f"SIMION validation failed: {e}.")
                
    print("\n==========================================")
    print("              FIN DE EJECUCIÓN            ")
    print("==========================================\n")

if __name__ == "__main__":
    main()
