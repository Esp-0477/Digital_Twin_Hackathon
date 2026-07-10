import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network that maps a 400-dimensional flattened beam histogram X
    to the mean and log-variance of the latent distribution q(z|X).
    """
    def __init__(self, input_dim=400, hidden_dims=[256, 128, 64], latent_dim=4):
        super(Encoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that maps a latent point z in R^d back to the
    400-dimensional reconstructed beam histogram X_hat.
    """
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
        
        # Output layer with Softmax to guarantee a true probability distribution (sum = 1)
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, z):
        features = self.decoder_layers(z)
        x_hat = self.fc_out(features)
        return x_hat


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for state compression.
    """
    def __init__(self, input_dim=400, latent_dim=4, encoder_hidden=[256, 128, 64], decoder_hidden=[64, 128, 256]):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, encoder_hidden, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden, input_dim)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode_to_latent(self, x):
        """
        Convenience function to get the latent mean vector (z) directly.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu


def vae_loss_fn(x_hat, x, mu, logvar, kl_beta=1.0):
    """
    Computes the VAE loss: Reconstruction Loss (Cross-Entropy) + KL Divergence.
    
    kl_beta: Scaling parameter for KL divergence (Beta-VAE concept to balance reconstruction vs. latent space smoothness)
    """
    # Reconstruction loss (Cross-Entropy for probability distributions)
    recon_loss = -torch.sum(x * torch.log(x_hat + 1e-8), dim=-1).mean()
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # averaged across batch dimensions
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = recon_loss + kl_beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
