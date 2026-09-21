import torch.nn as nn
import numpy as np
import torch

torch.manual_seed(42)

class encoder(nn.Module):
    """Encodes a 96x96 RGB image into the parameters of a Gaussian latent distribution."""
    def __init__(self):
        super().__init__()
        self.fc_mu = nn.Linear(128*6*6, 128)
        self.fc_logvar = nn.Linear(128*6*6, 128)
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, 3, 96, 96), pixel values in [-1, 1]
        Returns:
            mu: tensor of shape (batch, 128) - mean of the latent distribution
            logvar: tensor of shape (batch, 128) - log-variance of the latent distribution
        """
        x = self.network(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class decoder(nn.Module):
    """Decodes a 128-dim latent vector back into a 96x96 RGB image."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 128*6*6)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, 128) - a sampled latent vector z
        Returns:
            x_recon: tensor of shape (batch, 3, 96, 96), pixel values in [-1, 1]
        """
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(x.shape[0],128,6,6)
        x_recon = self.network(x)
        return x_recon

class VAE(nn.Module):
    """Variational Autoencoder combining the encoder, reparameterization, and decoder."""
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def reparametrize(self, mu, logvar):
        """
        Samples z from N(mu, exp(logvar)) using the reparameterization trick.

        Args:
            mu: tensor of shape (batch, 128)
            logvar: tensor of shape (batch, 128)
        Returns:
            z: tensor of shape (batch, 128)
        """
        # Sampling directly from N(mu, sigma^2) is non-differentiable, so gradients
        # can't flow back to mu/logvar. Instead we sample epsilon ~ N(0,1) (a constant,
        # untouched by autograd) and compute z = mu + std * epsilon, which is a
        # differentiable function of mu and std.
        return mu + torch.exp(0.5*logvar) * torch.randn_like(mu)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, 3, 96, 96), pixel values in [-1, 1]
        Returns:
            mu: tensor of shape (batch, 128)
            logvar: tensor of shape (batch, 128)
            x_recon: tensor of shape (batch, 3, 96, 96), reconstruction of x
        """
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return mu, logvar, x_recon


if __name__ == "__main__":
    model = VAE()
    sanity_check = torch.rand(4, 3, 96, 96)
    mu, logvar, x_recon = model(sanity_check)
    print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}, x_recon shape = {x_recon.shape}")    