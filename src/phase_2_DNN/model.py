import torch
import torch.nn as nn
import sys
import pathlib

# Add phase_1_VAE to path to import Decoder
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent / "phase_1_VAE"))

try:
    from vae import Decoder
except ImportError:
    # Fallback definition if import fails
    class Decoder(nn.Module):
        def __init__(self, latent_dim=2, hidden_dims=[64, 128, 256], output_dim=400):
            super(Decoder, self).__init__()
            layers = []
            in_dim = latent_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            self.decoder_layers = nn.Sequential(*layers)
            self.fc_out = nn.Sequential(nn.Linear(in_dim, output_dim), nn.Sigmoid())
        def forward(self, z):
            return self.fc_out(self.decoder_layers(z))


class ForwardRegressor(nn.Module):
    """
    DNN that maps normalized voltages y in R^8 to latent coordinates z in R^latent_dim.
    """
    def __init__(self, input_dim=8, hidden_dims=[64, 128, 64], latent_dim=2):
        super(ForwardRegressor, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, latent_dim)
        
    def forward(self, y):
        features = self.network(y)
        z = self.fc_out(features)
        return z


class FullEmulator(nn.Module):
    """
    Combined emulator: y (voltages) -> ForwardRegressor -> z_hat -> VAE Decoder -> X_hat (histogram).
    This combined pipeline is fully differentiable.
    """
    def __init__(self, forward_regressor: ForwardRegressor, decoder: Decoder):
        super(FullEmulator, self).__init__()
        self.forward_regressor = forward_regressor
        self.decoder = decoder
        
        # Freeze VAE Decoder parameters. We only train the ForwardRegressor in Phase 2.
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, y):
        # 1. Predict latent coordinates z_hat from voltages y
        z_hat = self.forward_regressor(y)
        # 2. Decode the latent coordinates to get reconstructed histogram X_hat
        x_hat = self.decoder(z_hat)
        return x_hat, z_hat
