import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pathlib
import sys
import numpy as np

# Add parent folders to path to import VAE and Dataset
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
sys.path.append(str(CURRENT_DIR.parent / "phase_1_VAE"))

from vae import VAE
from dataset import BeamlineDataset

class InverseEstimator(nn.Module):
    """
    DNN that maps latent coordinates z in R^latent_dim to normalized voltages y in R^8.
    This solves the Inverse Setpoint problem (Consigna Inversa).
    """
    def __init__(self, latent_dim=4, hidden_dims=[64, 128, 64], output_dim=8):
        super(InverseEstimator, self).__init__()
        
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        features = self.network(z)
        return self.fc_out(features)


def train_inverse_estimator(
    dataset_path: str,
    vae_weights_path: str,
    epochs=200,
    batch_size=8,
    learning_rate=1e-3,
    latent_dim=4,
    save_model_name="inverse_estimator.pt"
):
    print("=== Training Inverse Estimator (Latent Space -> Voltages) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load VAE
    vae_model = VAE(input_dim=400, latent_dim=latent_dim).to(device)
    try:
        vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
        vae_model.eval()
        print("Successfully loaded VAE weights.")
    except FileNotFoundError:
        print(f"Error: VAE weights not found at {vae_weights_path}")
        return
        
    # 2. Load primary dataset
    try:
        dataset = BeamlineDataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    # 3. Filter for trials with hits (transmission > 0)
    # If we include 0-hit trials, multiple different voltages map to the same zero histogram (latent point),
    # creating a one-to-many conflict that ruins training.
    raw_histograms = dataset.raw_histograms
    transmissions = np.sum(raw_histograms, axis=1) * 500
    valid_indices = np.where(transmissions > 0)[0]
    
    print(f"Total dataset trials: {len(dataset)}")
    print(f"Trials with hits (transmission > 0): {len(valid_indices)}")
    
    if len(valid_indices) < 5:
        print("Warning: Too few trials with hits. Consider running the optimizer longer to collect more hits.")
        # If there are extremely few hits, fall back to using all data (though results will be poorer)
        valid_indices = np.arange(len(dataset))
        
    # Encode only the valid histograms to latent coordinates (z)
    norm_voltages_list = []
    latents_list = []
    
    # Extract only valid samples from the BeamlineDataset
    for idx in valid_indices:
        voltages_tensor, histogram_tensor, _ = dataset[idx]
        
        # Add batch dimension and send to device
        histogram_tensor = histogram_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae_model.encoder(histogram_tensor)
            
        norm_voltages_list.append(voltages_tensor.cpu())
        latents_list.append(mu.squeeze(0).cpu())
        
    voltages_tensor = torch.stack(norm_voltages_list, dim=0) # shape: (N_hits, 8)
    latents_tensor = torch.stack(latents_list, dim=0)       # shape: (N_hits, latent_dim)
    
    # 4. Create TensorDataset
    inverse_dataset = TensorDataset(latents_tensor, voltages_tensor)
    
    # Split 80/20 train/validation
    dataset_size = len(inverse_dataset)
    val_size = max(1, int(dataset_size * 0.2))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(inverse_dataset, [train_size, val_size])
    
    # Avoid BatchNorm1d error when the last batch has size 1
    drop_last = (train_size % batch_size == 1) and train_size > 1
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
    
    # 5. Initialize model, optimizer, criterion
    estimator = InverseEstimator(latent_dim=latent_dim, output_dim=8).to(device)
    optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        estimator.train()
        train_loss = 0.0
        for batch_latents, batch_voltages in train_loader:
            batch_latents = batch_latents.to(device)
            batch_voltages = batch_voltages.to(device)
            
            preds = estimator(batch_latents)
            loss = criterion(preds, batch_voltages)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_latents.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        estimator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_latents, batch_voltages in val_loader:
                batch_latents = batch_latents.to(device)
                batch_voltages = batch_voltages.to(device)
                
                preds = estimator(batch_latents)
                loss = criterion(preds, batch_voltages)
                
                val_loss += loss.item() * batch_latents.size(0)
        val_loss /= len(val_loader.dataset)
        
        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
            
        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CURRENT_DIR / save_model_name
            torch.save(estimator.state_dict(), save_path)
            
    print(f"Training complete. Best Validation MSE: {best_val_loss:.6f}")
    print(f"Model saved to {CURRENT_DIR / save_model_name}")

if __name__ == "__main__":
    dataset_file = CURRENT_DIR.parent.parent / "Hackathon_student" / "beamline_dataset.npz"
    vae_weights = CURRENT_DIR.parent / "phase_1_VAE" / "vae_model.pt"
    train_inverse_estimator(str(dataset_file), str(vae_weights), epochs=200, batch_size=8, latent_dim=4)
