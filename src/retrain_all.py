import sys
import pathlib
import subprocess

# Add src directories to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR / "phase_1_VAE"))
sys.path.append(str(CURRENT_DIR / "phase_2_DNN"))
sys.path.append(str(CURRENT_DIR / "phase_3_control"))

def main():
    print("==========================================")
    # 1. Update dataset
    print("Step 1: Running LHS and Local Perturbations Data Collection...")
    collect_script = CURRENT_DIR / "phase_1_VAE" / "collect_all_real_data.py"
    subprocess.run([sys.executable, str(collect_script)], check=True)
    
    # Paths
    dataset_file = CURRENT_DIR.parent / "Hackathon_student" / "beamline_dataset.npz"
    vae_weights = CURRENT_DIR / "phase_1_VAE" / "vae_model.pt"
    
    # 2. Retrain VAE
    print("\nStep 2: Retraining Variational Autoencoder (VAE)...")
    from train_vae import train_vae
    train_vae(
        data_path=str(dataset_file),
        epochs=1200,      # Slightly more epochs for better learning
        batch_size=8,
        latent_dim=2,
        kl_beta=0.005,
        save_model_name="vae_model.pt"
    )
    
    # 3. Retrain Forward Regressor
    print("\nStep 3: Retraining Forward Regressor...")
    from train_dnn import train_forward_regressor
    train_forward_regressor(
        dataset_path=str(dataset_file),
        vae_weights_path=str(vae_weights),
        epochs=200,
        batch_size=8,
        latent_dim=2,
        save_regressor_name="forward_regressor.pt"
    )
    
    # 4. Train Inverse Estimator
    print("\nStep 4: Training Inverse Estimator...")
    from inverse_estimator import train_inverse_estimator
    train_inverse_estimator(
        dataset_path=str(dataset_file),
        vae_weights_path=str(vae_weights),
        epochs=250,
        batch_size=8,
        latent_dim=2,
        save_model_name="inverse_estimator.pt"
    )
    
    # 5. Generate Diagnostics Plots
    print("\nStep 5: Generating diagnostic plots...")
    plot_script = CURRENT_DIR / "generate_plots.py"
    subprocess.run([sys.executable, str(plot_script)], check=True)
    
    print("\n==========================================")
    print("All models successfully trained, updated, and plotted!")
    print("==========================================")

if __name__ == "__main__":
    main()
