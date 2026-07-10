import torch
import numpy as np
import pathlib
import sys
import os
from sklearn.neighbors import KernelDensity
import optuna

# Add directories to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
sys.path.append(str(CURRENT_DIR.parent / "phase_1_VAE"))
sys.path.append(str(CURRENT_DIR.parent / "phase_2_DNN"))

from vae import VAE
from model import ForwardRegressor, FullEmulator
from dataset import BeamlineDataset, run_single_simulation, build_histogram, OPTIMIZE, DETECTOR_REGION
from inverse_estimator import InverseEstimator

class LatentControlSystem:
    """
    Unified Latent Control System inspired by CBOL-Tuner.
    Integrates VAE, Forward Regressor, Inverse Estimator, and KDE Density Filter
    to solve Steering (Dirección) and Inverse Setpoint (Consigna Inversa) tasks.
    """
    def __init__(self, dataset_path, vae_weights, regressor_weights, inverse_weights, latent_dim=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        
        # Load dataset
        self.dataset = BeamlineDataset(dataset_path)
        
        # Load VAE
        self.vae = VAE(input_dim=400, latent_dim=latent_dim).to(self.device)
        self.vae.load_state_dict(torch.load(vae_weights, map_location=self.device))
        self.vae.eval()
        
        # Load Forward Regressor (y -> z)
        self.forward_reg = ForwardRegressor(input_dim=8, latent_dim=latent_dim).to(self.device)
        self.forward_reg.load_state_dict(torch.load(regressor_weights, map_location=self.device))
        self.forward_reg.eval()
        
        # Load Inverse Estimator (z -> y)
        self.inverse_est = InverseEstimator(latent_dim=latent_dim, output_dim=8).to(self.device)
        self.inverse_est.load_state_dict(torch.load(inverse_weights, map_location=self.device))
        self.inverse_est.eval()
        
        # Load Full Emulator (voltages y -> VAE Decoded Histogram X_hat)
        self.emulator = FullEmulator(self.forward_reg, self.vae.decoder).to(self.device)
        self.emulator.eval()
        
        # Train Density Filter on Latent Space
        self._fit_density_filter()

    def _fit_density_filter(self):
        """
        Fits a Kernel Density Estimator (KDE) on the latent vectors of the training data
        to prune out hallucinated states (Admissibility / Rubric G).
        """
        # Encode all training histograms to z
        histograms_t = torch.tensor(self.dataset.histograms, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            true_z, _ = self.vae.encoder(histograms_t)
            true_z = true_z.cpu().numpy()
            
        self.train_z = true_z
        self.z_min = true_z.min(axis=0)
        self.z_max = true_z.max(axis=0)
        
        # Fit KDE
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(true_z)
        
        # Determine anomaly threshold: 5th percentile of training log-likelihoods
        log_densities = self.kde.score_samples(true_z)
        self.threshold = np.percentile(log_densities, 5)
        print(f"KDE Density Filter fitted. Latent bounds: z1 [{self.z_min[0]:.2f}, {self.z_max[0]:.2f}], z2 [{self.z_min[1]:.2f}, {self.z_max[1]:.2f}]")
        print(f"KDE 5th percentile log-likelihood threshold: {self.threshold:.4f}")

    def is_physical_state(self, z):
        """
        Checks if a latent point z belongs to the physical manifold using the KDE Density Filter.
        """
        z_np = np.array(z, dtype=np.float32).reshape(1, -1)
        log_density = self.kde.score_samples(z_np)[0]
        return log_density >= self.threshold, log_density

    def decode_latent(self, z):
        """
        Decodes a latent vector z to a beam profile histogram X_hat.
        """
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            x_hat = self.vae.decoder(z_t)
        return x_hat.squeeze(0).cpu()

    def score_profile(self, x_hat, objective="hits"):
        """
        Evaluates the quality of a decoded beam profile.
          - "hits": total predicted transmission (sum of normalized bins)
          - "spread": beam width (standard deviation of arrival distribution)
        """
        if objective == "hits":
            # Sum of bins is transmission fraction [0, 1]
            return float(torch.sum(x_hat).item())
        else:
            # Spread: standard deviation in X and Y
            grid = x_hat.view(20, 20)
            y_coords, x_coords = torch.meshgrid(torch.arange(20), torch.arange(20), indexing='ij')
            total_intensity = torch.sum(grid)
            if total_intensity <= 1e-5:
                return 1e9  # Penalty for no beam
            
            mean_y = torch.sum(grid * y_coords) / total_intensity
            mean_x = torch.sum(grid * x_coords) / total_intensity
            
            var_y = torch.sum(grid * (y_coords - mean_y)**2) / total_intensity
            var_x = torch.sum(grid * (x_coords - mean_x)**2) / total_intensity
            
            spread = float((torch.sqrt(var_y) + torch.sqrt(var_x)).item())
            return spread

    def refine_voltages_with_backprop(self, initial_norm_voltages, target_histogram=None, objective="hits", steps=60, lr=0.01, alpha=2.0):
        """
        Refines normalized voltages using backpropagation through the differentiable emulator,
        with a proximal L2 regularization term to keep the voltages close to the initial guess
        and avoid out-of-distribution exploitation.
        """
        y_init_t = torch.tensor(initial_norm_voltages, dtype=torch.float32, device=self.device)
        y_param = torch.tensor(initial_norm_voltages, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([y_param], lr=lr)
        
        if target_histogram is not None:
            target_t = torch.tensor(target_histogram, dtype=torch.float32, device=self.device)
            criterion = torch.nn.MSELoss()
            
        for step in range(steps):
            optimizer.zero_grad()
            
            # Predict histogram via FullEmulator
            x_hat, _ = self.emulator(y_param.unsqueeze(0))
            x_hat = x_hat.squeeze(0)
            
            if target_histogram is not None:
                loss_term = criterion(x_hat, target_t)
            else:
                # Maximize transmission = minimize negative sum of histogram
                loss_term = -torch.sum(x_hat)
                
            # L2 regularization term to keep refined voltages near physical manifold
            reg_term = alpha * torch.sum((y_param - y_init_t)**2)
            
            loss = loss_term + reg_term
            loss.backward()
            optimizer.step()
            
            # Clamp voltages to keep them inside [-1, 1] (physical bounds)
            with torch.no_grad():
                y_param.clamp_(-1.0, 1.0)
                
        refined_norm_voltages = y_param.detach().cpu().numpy()
        return refined_norm_voltages

    def optimize_latent_steering(self, objective="hits", n_trials=100):
        """
        Classifier-Pruned Latent Bayesian Optimization (C-BO) refined by Gradient Descent.
        Searches the 2D latent space for optimal beam and prunes anomalies,
        then fine-tunes the resulting voltages using backpropagation.
        """
        print(f"\n--- Running Latent Steering Optimization ({objective}) ---")
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "maximize" if objective == "hits" else "minimize"
        study = optuna.create_study(direction=direction)
        
        # Latent space search box (expanded 10% of range beyond training bounds)
        z1_range = max(1e-5, self.z_max[0] - self.z_min[0])
        z2_range = max(1e-5, self.z_max[1] - self.z_min[1])
        z1_min, z1_max = self.z_min[0] - 0.1 * z1_range, self.z_max[0] + 0.1 * z1_range
        z2_min, z2_max = self.z_min[1] - 0.1 * z2_range, self.z_max[1] + 0.1 * z2_range
        
        def optuna_obj(trial):
            z1 = trial.suggest_float("z1", z1_min, z1_max)
            z2 = trial.suggest_float("z2", z2_min, z2_max)
            z = [z1, z2]
            
            # 1. Apply KDE Density Filter (Classifier-Pruning)
            is_valid, _ = self.is_physical_state(z)
            if not is_valid:
                return -1.0 if objective == "hits" else 1e9
                
            # 2. Decode state and score
            x_hat = self.decode_latent(z)
            return self.score_profile(x_hat, objective)

        study.optimize(optuna_obj, n_trials=n_trials)
        best_z = [study.best_params["z1"], study.best_params["z2"]]
        print(f"Optimal Latent State found: z* = {best_z}")
        print(f"Predicted latent score: {study.best_value:.4f}")
        
        # Map z* to normalized voltages using the Inverse Estimator (initial rough guess)
        best_z_t = torch.tensor(best_z, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            norm_voltages_init = self.inverse_est(best_z_t).squeeze(0).cpu().numpy()
            
        print(f"Initial voltages from Inverse Estimator (rough guess): " + 
              ", ".join(f"{v:+.2f}" for v in norm_voltages_init))
              
        # Refine voltages using the differentiable FullEmulator backpropagation
        print("Refining voltages via Backpropagation...")
        norm_voltages_refined = self.refine_voltages_with_backprop(norm_voltages_init, objective=objective, steps=80, lr=0.02)
        print(f"Refined voltages (gradients matched):               " + 
              ", ".join(f"{v:+.2f}" for v in norm_voltages_refined))
              
        # Denormalize to physical voltages
        physical_voltages = self.dataset.denormalize_voltages(norm_voltages_refined)
        
        # Format as electrode dict
        opt_keys = sorted(list(OPTIMIZE.keys()))
        voltage_dict = {k: float(v) for k, v in zip(opt_keys, physical_voltages)}
        
        return best_z, voltage_dict

    def run_inverse_setpoint(self, target_histogram):
        """
        Solves the Inverse Setpoint problem (Consigna Inversa) refined by Gradient Descent.
        Given a target histogram, maps to latent z_target, infers initial voltages,
        and uses backpropagation on the Forward Emulator to match the target profile.
        """
        print("\n--- Running Inverse Setpoint (Consigna Inversa) ---")
        # 1. Convert to tensor
        target_t = torch.tensor(target_histogram, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. Encode to latent space
        with torch.no_grad():
            z_target, _ = self.vae.encoder(target_t)
            z_target = z_target.squeeze(0).cpu().numpy()
            
        print(f"Target profile maps to latent coordinates: z_target = {z_target}")
        
        # 3. Check density
        is_valid, log_dens = self.is_physical_state(z_target)
        if not is_valid:
            print(f"Warning: z_target is in a low-density region (log-likelihood: {log_dens:.2f} < {self.threshold:.2f}).")
            # Project to closest point in training latents
            dists = np.sum((self.train_z - z_target)**2, axis=1)
            closest_idx = np.argmin(dists)
            z_target = self.train_z[closest_idx]
            print(f"Projected target to closest training point: z_target = {z_target}")
            
        # 4. Map z_target to normalized voltages using Inverse Estimator (initial rough guess)
        z_target_t = torch.tensor(z_target, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            norm_voltages_init = self.inverse_est(z_target_t).squeeze(0).cpu().numpy()
            
        print(f"Initial voltages from Inverse Estimator (rough guess): " + 
              ", ".join(f"{v:+.2f}" for v in norm_voltages_init))
              
        # 5. Refine voltages using differentiable emulator backprop to reconstruct target profile
        print("Refining voltages via Backpropagation (matching profile)...")
        norm_voltages_refined = self.refine_voltages_with_backprop(
            norm_voltages_init, target_histogram=target_histogram, steps=100, lr=0.02
        )
        print(f"Refined voltages (profile matched):                 " + 
              ", ".join(f"{v:+.2f}" for v in norm_voltages_refined))
              
        # 6. Denormalize to physical voltages
        physical_voltages = self.dataset.denormalize_voltages(norm_voltages_refined)
        
        # Format as electrode dict
        opt_keys = sorted(list(OPTIMIZE.keys()))
        voltage_dict = {k: float(v) for k, v in zip(opt_keys, physical_voltages)}
        
        return z_target.tolist(), voltage_dict

    def validate_in_simion(self, voltages):
        """
        Runs the real SIMION simulator with the given voltages to check the true score.
        """
        print(f"Running SIMION validation...")
        print(f"Voltages: " + ", ".join(f"V{k}={v:.2f}" for k, v in sorted(voltages.items())))
        
        positions = run_single_simulation(voltages)
        hist = build_histogram(positions)
        hits = int(np.sum(hist) * 500)
        
        # Calculate spread
        x_min, x_max = DETECTOR_REGION["x"]
        y_min, y_max = DETECTOR_REGION["y"]
        z_min, z_max = DETECTOR_REGION["z"]
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
        valid_pts = positions[mask]
        
        if len(valid_pts) > 0:
            spread = float(np.std(valid_pts[:, 1]) + np.std(valid_pts[:, 2]))
        else:
            spread = 1e9
            
        print(f"SIMION Results -> True Hits: {hits}/500 ({hits/5.0:.1f}%), True Spread: {spread:.3f}")
        return hist, hits, spread


def test_control_pipeline():
    print("=== Testing Latent Control Pipeline ===")
    
    # 1. File paths
    dataset_file = CURRENT_DIR.parent.parent / "Hackathon_student" / "beamline_dataset.npz"
    vae_weights = CURRENT_DIR.parent / "phase_1_VAE" / "vae_model.pt"
    regressor_weights = CURRENT_DIR.parent / "phase_2_DNN" / "forward_regressor.pt"
    inverse_weights = CURRENT_DIR / "inverse_estimator.pt"
    
    # Check that weights exist
    for p in [dataset_file, vae_weights, regressor_weights, inverse_weights]:
        if not p.exists():
            print(f"Error: Required file not found: {p}")
            return

    # Instantiate Control System
    lcs = LatentControlSystem(
        dataset_path=str(dataset_file),
        vae_weights=str(vae_weights),
        regressor_weights=str(regressor_weights),
        inverse_weights=str(inverse_weights)
    )
    
    # =========================================================================
    # Task 1: Steering (Dirección)
    # =========================================================================
    # Find voltages that maximize transmission (hits)
    best_z, opt_voltages = lcs.optimize_latent_steering(objective="hits", n_trials=150)
    
    # Validate in SIMION
    true_hist, true_hits, true_spread = lcs.validate_in_simion(opt_voltages)
    print(f"Latent steering validation complete.")
    
    # =========================================================================
    # Task 2: Inverse Setpoint (Consigna Inversa)
    # =========================================================================
    # Let's extract a target histogram from the training data that has good transmission
    raw_histograms = lcs.dataset.raw_histograms
    transmissions = np.sum(raw_histograms, axis=1) * 500
    best_train_idx = np.argmax(transmissions)
    target_profile = raw_histograms[best_train_idx]
    target_hits = int(transmissions[best_train_idx])
    
    print(f"\nSetting target profile to Trial {best_train_idx} (Hits: {target_hits})")
    
    z_target, inferred_voltages = lcs.run_inverse_setpoint(target_profile)
    
    # Validate in SIMION
    true_recon_hist, recon_hits, recon_spread = lcs.validate_in_simion(inferred_voltages)
    print(f"Inverse setpoint validation complete. Target: {target_hits} hits, Reconstructed: {recon_hits} hits.")
    
    # Print comparison
    print("\n=== Voltage Comparison ===")
    opt_keys = sorted(list(OPTIMIZE.keys()))
    original_voltages = lcs.dataset.raw_voltages[best_train_idx]
    
    print("Electrode | Original Volts | Inferred Volts | Diff")
    print("-" * 50)
    for idx, key in enumerate(opt_keys):
        orig = original_voltages[idx]
        inf = inferred_voltages[key]
        diff = inf - orig
        print(f"  V{key:<6} | {orig:14.2f} | {inf:14.2f} | {diff:+7.2f}")


if __name__ == "__main__":
    test_control_pipeline()
