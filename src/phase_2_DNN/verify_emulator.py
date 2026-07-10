import torch
import sys
import pathlib

# Add src directories to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent / "phase_1_VAE"))

from model import ForwardRegressor, FullEmulator
from vae import VAE

def verify():
    print("=== Verifying Full Emulator Differentiability ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Paths
    vae_weights_path = CURRENT_DIR.parent / "phase_1_VAE" / "vae_model.pt"
    regressor_weights_path = CURRENT_DIR / "forward_regressor.pt"
    
    # 2. Instantiate and load VAE
    vae_model = VAE(input_dim=400, latent_dim=2).to(device)
    try:
        vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
        print("Loaded VAE model weights.")
    except FileNotFoundError:
        print(f"Error: VAE weights not found at {vae_weights_path}. Please train the VAE first.")
        return
        
    # 3. Instantiate and load ForwardRegressor
    regressor = ForwardRegressor(input_dim=8, latent_dim=2).to(device)
    try:
        regressor.load_state_dict(torch.load(regressor_weights_path, map_location=device))
        print("Loaded Forward Regressor weights.")
    except FileNotFoundError:
        print(f"Error: Regressor weights not found at {regressor_weights_path}. Please train the Regressor first.")
        return
        
    # 4. Construct Full Emulator
    emulator = FullEmulator(regressor, vae_model.decoder).to(device)
    emulator.eval()
    
    # 5. Define a mock input voltage setting
    # In normalized range [-1, 1]
    # Let's set V3=0.5, V6=-0.2, V9=0.8, V10=-0.5, V11=0.0, V12=-0.9, V15=0.1, V18=0.4
    voltages = torch.tensor([[0.5, -0.2, 0.8, -0.5, 0.0, -0.9, 0.1, 0.4]], 
                            dtype=torch.float32, 
                            device=device, 
                            requires_grad=True)
    
    # 6. Forward Pass
    x_hat, z_hat = emulator(voltages)
    
    # 7. Compute predicted Transmission (sum of normalized histogram bins)
    # x_hat is of shape (1, 400). Its elements represent normalized bin values.
    # Summing them up yields the total transmission fraction (between 0.0 and 1.0)
    transmission_score = torch.sum(x_hat)
    
    # 8. Backward Pass to get gradients
    # We want to see: d(transmission) / d(voltages)
    transmission_score.backward()
    
    # 9. Report Results
    print("\n--- Emulator Output ---")
    print(f"Predicted Latent Space Coordinates z: {z_hat[0].detach().cpu().numpy()}")
    print(f"Predicted Transmission Score:         {transmission_score.item():.4f} ({transmission_score.item() * 100:.2f}%)")
    
    print("\n--- Analytical Gradients (d_transmission / d_voltages) ---")
    gradients = voltages.grad[0].cpu().numpy()
    electrode_labels = ["V3", "V6", "V9", "V10", "V11", "V12", "V15", "V18"]
    for label, grad in zip(electrode_labels, gradients):
        print(f"   {label}: {grad:+.6f}")
        
    print("\nVerification successful! The model is fully differentiable from inputs to outputs.")
    print("These analytical gradients can be used to optimize voltages instantly using gradient descent.")

if __name__ == "__main__":
    verify()
