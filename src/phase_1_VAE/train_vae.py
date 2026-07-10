import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pathlib
import sys
import numpy as np

# Add src/phase_1_VAE folder to path
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from vae import VAE, vae_loss_fn
from dataset import BeamlineDataset

def train_vae(
    data_path: str,
    epochs=100,
    batch_size=8,
    learning_rate=1e-3,
    latent_dim=2,
    kl_beta=0.1,
    save_model_name="vae_model.pt"
):
    print("=== Training Variational Autoencoder (VAE) ===")
    
    # Load dataset
    try:
        dataset = BeamlineDataset(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset file {data_path} not found. Please run collect_data.py first.")
        return
        
    dataset_size = len(dataset)
    print(f"Dataset loaded. Total samples: {dataset_size}")
    
    if dataset_size < 5:
        print("Warning: Dataset is very small. Training might overfit or fail.")
        
    # Split into train and validation sets (80% train, 20% val)
    val_size = max(1, int(dataset_size * 0.2))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Balance training dataset by oversampling positive hit samples to avoid majority class collapse
    train_indices = train_dataset.indices
    train_histograms = dataset.histograms[train_indices]
    train_transmissions = train_histograms.sum(axis=1) * 500
    has_hits_idx = np.where(train_transmissions > 0)[0]
    no_hits_idx = np.where(train_transmissions == 0)[0]
    
    if len(has_hits_idx) > 0:
        num_repeats = int(len(no_hits_idx) / len(has_hits_idx)) + 1
        repeated_hits = np.repeat(has_hits_idx, num_repeats)
        balanced_train_subindices = np.concatenate([no_hits_idx, repeated_hits])
        balanced_train_indices = [train_indices[i] for i in balanced_train_subindices]
    else:
        balanced_train_indices = train_indices
        
    from torch.utils.data import Subset
    balanced_train_dataset = Subset(dataset, balanced_train_indices)
    
    # Data loaders
    train_loader = DataLoader(balanced_train_dataset, batch_size=min(batch_size, len(balanced_train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VAE(input_dim=400, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for _, histograms in train_loader:
            histograms = histograms.to(device)
            
            # Forward pass
            x_hat, mu, logvar = model(histograms)
            
            # Binary Cross Entropy loss (BCE) to prevent posterior collapse
            import torch.nn.functional as F
            recon_loss = F.binary_cross_entropy(x_hat, histograms, reduction='sum') / histograms.size(0)
            
            # KL loss
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + kl_beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * histograms.size(0)
            train_recon += recon_loss.item() * histograms.size(0)
            train_kl += kl_loss.item() * histograms.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_recon /= len(train_loader.dataset)
        train_kl /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for _, histograms in val_loader:
                histograms = histograms.to(device)
                x_hat, mu, logvar = model(histograms)
                
                # Use same BCE loss structure for validation monitoring
                import torch.nn.functional as F
                recon_loss = F.binary_cross_entropy(x_hat, histograms, reduction='sum') / histograms.size(0)
                
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_loss + kl_beta * kl_loss
                
                val_loss += loss.item() * histograms.size(0)
                val_recon += recon_loss.item() * histograms.size(0)
                val_kl += kl_loss.item() * histograms.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_kl /= len(val_loader.dataset)
        
        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train Loss: {train_loss:.6f} (Weighted Recon: {train_recon:.6f}, KL: {train_kl:.6f}) | "
                  f"Val Loss: {val_loss:.6f} (Weighted Recon: {val_recon:.6f}, KL: {val_kl:.6f})")
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = CURRENT_DIR / save_model_name
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Training completed. Best Validation Loss: {best_val_loss:.6f}")
    print(f"Best model weights saved to {CURRENT_DIR / save_model_name}")

if __name__ == "__main__":
    dataset_file = CURRENT_DIR.parent.parent / "Hackathon_student" / "beamline_dataset.npz"
    train_vae(str(dataset_file), epochs=1000, batch_size=8, latent_dim=2, kl_beta=0.005)
