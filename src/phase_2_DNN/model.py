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
        def __init__(self, latent_dim=4, hidden_dims=[64, 128, 256], output_dim=400):
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
    def __init__(self, input_dim=8, hidden_dims=[64, 128, 64], latent_dim=4):
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


class TransmissionRegressor(nn.Module):
    """
    DNN that maps normalized voltages y in R^8 to predicted transmission fraction T in [0, 1].
    """
    def __init__(self, input_dim=8, hidden_dims=[64, 128, 64]):
        super(TransmissionRegressor, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid() # Salida estrictamente en [0, 1]
        )
        
    def forward(self, y):
        features = self.network(y)
        return self.fc_out(features).squeeze(-1)


class FullEmulator(nn.Module):
    """
    Combined emulator: y (voltages) -> ForwardRegressor -> z_hat -> VAE Decoder -> X_hat (histogram shape).
    Optionally scales the shape by the predicted transmission from TransmissionRegressor.
    This combined pipeline is fully differentiable.
    """
    def __init__(self, forward_regressor: ForwardRegressor, decoder: Decoder, transmission_regressor: nn.Module = None):
        super(FullEmulator, self).__init__()
        self.forward_regressor = forward_regressor
        self.decoder = decoder
        self.transmission_regressor = transmission_regressor
        
        # Freeze VAE Decoder parameters. We only train the ForwardRegressor in Phase 2.
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # Freeze Transmission Regressor parameters if provided during full emulation.
        if self.transmission_regressor is not None:
            for param in self.transmission_regressor.parameters():
                param.requires_grad = False
            
    def forward(self, y):
        # 1. Predict latent coordinates z_hat from voltages y
        z_hat = self.forward_regressor(y)
        # 2. Decode the latent coordinates to get reconstructed shape
        x_hat = self.decoder(z_hat)
        
        # 3. Scale shape by predicted transmission if available
        if self.transmission_regressor is not None:
            T_hat = self.transmission_regressor(y).unsqueeze(-1) # shape: (batch, 1)
            x_hat = T_hat * x_hat
            
        return x_hat, z_hat
