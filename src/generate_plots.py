import os
import sys
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add directories to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR / "phase_1_VAE"))
sys.path.append(str(CURRENT_DIR / "phase_2_DNN"))

from vae import VAE
from model import ForwardRegressor
from dataset import BeamlineDataset

def main():
    print("=== Generating Diagnostic Figures ===")
    
    # 1. Paths
    dataset_path = CURRENT_DIR.parent / "Hackathon_student" / "beamline_dataset.npz"
    vae_weights_path = CURRENT_DIR / "phase_1_VAE" / "vae_model.pt"
    regressor_weights_path = CURRENT_DIR / "phase_2_DNN" / "forward_regressor.pt"
    fig_dir = CURRENT_DIR.parent / "Figuras"
    
    os.makedirs(fig_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = BeamlineDataset(str(dataset_path))
    raw_histograms = dataset.raw_histograms # (N, 400)
    raw_voltages = dataset.raw_voltages     # (N, 8)
    
    # Load VAE
    vae = VAE(input_dim=400, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    vae.eval()
    
    # Load Forward Regressor
    regressor = ForwardRegressor(input_dim=8, latent_dim=2).to(device)
    regressor.load_state_dict(torch.load(regressor_weights_path, map_location=device))
    regressor.eval()
    
    # Get PyTorch tensors
    voltages_t = torch.tensor(dataset.voltages, dtype=torch.float32, device=device)
    histograms_t = torch.tensor(dataset.histograms, dtype=torch.float32, device=device)
    
    # Encode and predict
    with torch.no_grad():
        # True latents (from VAE Encoder)
        true_z, _ = vae.encoder(histograms_t)
        true_z = true_z.cpu().numpy()
        
        # Reconstructed histograms
        reconstructed_histograms_t, _, _ = vae(histograms_t)
        reconstructed_histograms = reconstructed_histograms_t.cpu().numpy()
        
        # Predicted latents (from DNN Regressor)
        pred_z = regressor(voltages_t).cpu().numpy()
        
    print("Models loaded and evaluations complete.")
    
    # Calculate transmission (number of hits = sum of bin values * 500)
    transmissions = np.sum(raw_histograms, axis=1) * 500
    
    # -------------------------------------------------------------------------
    # Plot 1: VAE Reconstruction (Real vs. Reconstructed)
    # -------------------------------------------------------------------------
    # Let's find a sample with hits and one without
    has_hits_idx = np.where(transmissions > 0)[0]
    
    if len(has_hits_idx) > 0:
        sample_idx_hits = has_hits_idx[0]
    else:
        sample_idx_hits = 0 # fallback
        
    # Get a sample with 0 hits (if any)
    no_hits_idx = np.where(transmissions == 0)[0]
    sample_idx_nohits = no_hits_idx[0] if len(no_hits_idx) > 0 else 1
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Sample with hits
    real_hits_grid = raw_histograms[sample_idx_hits].reshape(20, 20) * 500 # back to counts
    recon_hits_grid = reconstructed_histograms[sample_idx_hits].reshape(20, 20) * 500
    
    im1 = axes[0].imshow(real_hits_grid, cmap='viridis', origin='lower')
    axes[0].set_title(f"Real Histogram (Trial {sample_idx_hits}, Hits: {transmissions[sample_idx_hits]:.0f})")
    fig.colorbar(im1, ax=axes[0], label="Ion count")
    
    im2 = axes[1].imshow(recon_hits_grid, cmap='viridis', origin='lower')
    axes[1].set_title("VAE Reconstructed Histogram")
    fig.colorbar(im2, ax=axes[1], label="Reconstructed intensity")
    
    plt.tight_layout()
    plt.savefig(fig_dir / "vae_reconstruction.png", dpi=150)
    plt.close()
    print(f"Saved: {fig_dir / 'vae_reconstruction.png'}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Latent Space Scatter
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Plot points with 0 hits as grey dots, and points with hits colored
    mask_hits = transmissions > 0
    mask_nohits = transmissions == 0
    
    plt.scatter(true_z[mask_nohits, 0], true_z[mask_nohits, 1], c='lightgrey', label='0 Hits', alpha=0.8, edgecolors='grey', s=40)
    
    if np.any(mask_hits):
        sc = plt.scatter(true_z[mask_hits, 0], true_z[mask_hits, 1], c=transmissions[mask_hits], 
                         cmap='autumn_r', label='Hits > 0', alpha=1.0, edgecolors='black', s=80)
        plt.colorbar(sc, label="Number of hits")
        
    plt.title("2D Latent Space Projection (VAE Encoder)")
    plt.xlabel("Latent variable z_1")
    plt.ylabel("Latent variable z_2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "latent_space_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {fig_dir / 'latent_space_scatter.png'}")
    
    # -------------------------------------------------------------------------
    # Plot 3: DNN vs VAE Latent Space Predictions (Regressor Accuracy)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # z_1 true vs pred
    axes[0].scatter(true_z[:, 0], pred_z[:, 0], c='dodgerblue', alpha=0.8, edgecolors='darkblue', s=45)
    # Draw y=x line
    min_z1 = min(true_z[:, 0].min(), pred_z[:, 0].min()) - 0.2
    max_z1 = max(true_z[:, 0].max(), pred_z[:, 0].max()) + 0.2
    axes[0].plot([min_z1, max_z1], [min_z1, max_z1], 'r--', label='Perfect match')
    axes[0].set_title("Latent Coordinate z_1: True vs. Predicted")
    axes[0].set_xlabel("True z_1 (VAE Encoder)")
    axes[0].set_ylabel("Predicted z_1 (DNN Regressor)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # z_2 true vs pred
    axes[1].scatter(true_z[:, 1], pred_z[:, 1], c='emerald' if 'emerald' in plt.colormaps() else 'forestgreen', alpha=0.8, edgecolors='darkgreen', s=45)
    # Draw y=x line
    min_z2 = min(true_z[:, 1].min(), pred_z[:, 1].min()) - 0.2
    max_z2 = max(true_z[:, 1].max(), pred_z[:, 1].max()) + 0.2
    axes[1].plot([min_z2, max_z2], [min_z2, max_z2], 'r--', label='Perfect match')
    axes[1].set_title("Latent Coordinate z_2: True vs. Predicted")
    axes[1].set_xlabel("True z_2 (VAE Encoder)")
    axes[1].set_ylabel("Predicted z_2 (DNN Regressor)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "dnn_vs_vae_latent.png", dpi=150)
    plt.close()
    print(f"Saved: {fig_dir / 'dnn_vs_vae_latent.png'}")
    
    print("All diagnostic figures generated successfully inside Figuras directory.")

if __name__ == "__main__":
    main()
