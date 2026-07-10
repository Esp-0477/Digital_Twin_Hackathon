import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pathlib
import sys
import numpy as np

# Add src directories to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent / "phase_1_VAE"))

from model import ForwardRegressor
from vae import VAE
from dataset import BeamlineDataset

def train_forward_regressor(
    dataset_path: str,
    vae_weights_path: str,
    epochs=150,
    batch_size=8,
    learning_rate=1e-3,
    latent_dim=4,
    save_regressor_name="forward_regressor.pt"
):
    print("=== Training Forward Regressor (Voltages -> Latent Space) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load VAE model and weights
    vae_model = VAE(input_dim=400, latent_dim=latent_dim).to(device)
    try:
        vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
        vae_model.eval()
        print(f"Successfully loaded trained VAE from {vae_weights_path}")
    except FileNotFoundError:
        print(f"Error: Trained VAE weights file not found at {vae_weights_path}")
        return
        
    # 2. Load primary dataset
    try:
        dataset = BeamlineDataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    # 3. Filter for trials with hits (transmission > 0)
    has_hits_idx = np.where(dataset.transmissions > 0.0)[0]
    print(f"Total dataset trials: {len(dataset)}")
    print(f"Trials with hits (transmission > 0) for Latent Mapping: {len(has_hits_idx)}")
    
    if len(has_hits_idx) < 5:
        print("Warning: Too few trials with hits. Latent mapping training might overfit.")
        
    from torch.utils.data import Subset
    hit_dataset = Subset(dataset, has_hits_idx)
    
    # 4. Encode all valid histograms into their latent space representations (z)
    all_voltages = []
    all_latents = []
    
    loader = DataLoader(hit_dataset, batch_size=len(hit_dataset), shuffle=False)
    with torch.no_grad():
        for voltages, histograms, _ in loader:
            voltages = voltages.to(device)
            histograms = histograms.to(device)
            
            # Encode histograms to latent means (mu)
            mu, _ = vae_model.encoder(histograms)
            
            all_voltages.append(voltages.cpu())
            all_latents.append(mu.cpu())
            
    voltages_tensor = torch.cat(all_voltages, dim=0) # shape: (N_hits, 8)
    latents_tensor = torch.cat(all_latents, dim=0)   # shape: (N_hits, latent_dim)
    
    print(f"Successfully encoded {len(voltages_tensor)} samples to latent space.")
    
    # 5. Create new TensorDataset for voltages -> latents mapping
    latent_mapping_dataset = TensorDataset(voltages_tensor, latents_tensor)
    
    # Split into train/validation sets (80/20)
    dataset_size = len(latent_mapping_dataset)
    val_size = max(1, int(dataset_size * 0.2))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(latent_mapping_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
    
    # 6. Initialize ForwardRegressor model
    regressor = ForwardRegressor(input_dim=8, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # 7. Training Loop
    for epoch in range(1, epochs + 1):
        regressor.train()
        train_loss = 0.0
        for batch_voltages, batch_latents in train_loader:
            batch_voltages = batch_voltages.to(device)
            batch_latents = batch_latents.to(device)
            
            # Forward pass
            preds = regressor(batch_voltages)
            loss = criterion(preds, batch_latents)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_voltages.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_voltages, batch_latents in val_loader:
                batch_voltages = batch_voltages.to(device)
                batch_latents = batch_latents.to(device)
                preds = regressor(batch_voltages)
                loss = criterion(preds, batch_latents)
                val_loss += loss.item() * batch_voltages.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE Loss: {train_loss:.6f} | Val MSE Loss: {val_loss:.6f}")
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CURRENT_DIR / save_regressor_name
            torch.save(regressor.state_dict(), save_path)
            
    print(f"Training completed. Best Validation MSE Loss: {best_val_loss:.6f}")
    print(f"Best regressor weights saved to {CURRENT_DIR / save_regressor_name}")


def train_transmission_regressor(
    dataset_path: str,
    epochs=150,
    batch_size=8,
    learning_rate=1e-3,
    save_regressor_name="transmission_regressor.pt"
):
    print("=== Training Transmission Regressor (Voltages -> Transmission) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load primary dataset
    try:
        dataset = BeamlineDataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    # Split into train/validation sets (80/20) on ALL samples (both zero and non-zero transmission)
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * 0.2))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Balance training dataset by oversampling positive transmission samples
    train_indices = train_dataset.indices
    train_transmissions = dataset.transmissions[train_indices]
    has_hits_idx = np.where(train_transmissions > 0.0)[0]
    no_hits_idx = np.where(train_transmissions == 0.0)[0]
    
    if len(has_hits_idx) > 0:
        # Oversample positive hit samples to match no-hit samples count
        num_repeats = int(len(no_hits_idx) / len(has_hits_idx)) + 1
        repeated_hits = np.repeat(has_hits_idx, num_repeats)
        balanced_train_subindices = np.concatenate([no_hits_idx, repeated_hits])
        balanced_train_indices = [train_indices[i] for i in balanced_train_subindices]
    else:
        balanced_train_indices = train_indices
        
    from torch.utils.data import Subset
    balanced_train_dataset = Subset(dataset, balanced_train_indices)
    
    train_loader = DataLoader(balanced_train_dataset, batch_size=min(batch_size, len(balanced_train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
    
    # Initialize TransmissionRegressor model
    from model import TransmissionRegressor
    regressor = TransmissionRegressor(input_dim=8).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        regressor.train()
        train_loss = 0.0
        for batch_voltages, _, batch_transmissions in train_loader:
            batch_voltages = batch_voltages.to(device)
            batch_transmissions = batch_transmissions.to(device)
            
            # Forward pass
            preds = regressor(batch_voltages)
            loss = criterion(preds, batch_transmissions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_voltages.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_voltages, _, batch_transmissions in val_loader:
                batch_voltages = batch_voltages.to(device)
                batch_transmissions = batch_transmissions.to(device)
                preds = regressor(batch_voltages)
                loss = criterion(preds, batch_transmissions)
                val_loss += loss.item() * batch_voltages.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE Loss: {train_loss:.6f} | Val MSE Loss: {val_loss:.6f}")
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CURRENT_DIR / save_regressor_name
            torch.save(regressor.state_dict(), save_path)
            
    print(f"Training completed. Best Validation MSE Loss: {best_val_loss:.6f}")
    print(f"Best transmission regressor weights saved to {CURRENT_DIR / save_regressor_name}")


if __name__ == "__main__":
    dataset_file = CURRENT_DIR.parent.parent / "Hackathon_student" / "beamline_dataset.npz"
    vae_weights = CURRENT_DIR.parent / "phase_1_VAE" / "vae_model.pt"
    train_forward_regressor(str(dataset_file), str(vae_weights), epochs=150, batch_size=8, latent_dim=4)
    train_transmission_regressor(str(dataset_file), epochs=150, batch_size=8)
