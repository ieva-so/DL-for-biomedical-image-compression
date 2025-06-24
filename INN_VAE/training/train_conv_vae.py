import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ConvVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.feature_dim = 128 * 32 * 32
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 32, 32)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss_fn(recon_x, x, mu, logvar, z_recon, z_target, beta=0.01, lambda_z=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    z_loss = F.mse_loss(z_recon, z_target, reduction='sum')
    return recon_loss + beta * kld + lambda_z * z_loss

def train_conv_vae(model, dataloader, epochs=10, lr=1e-3, device="cpu", z_targets=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i, (batch, _) in enumerate(dataloader):
            batch = batch.to(device)
            z_target = z_targets[i * batch.size(0):(i + 1) * batch.size(0)].to(device) if z_targets is not None else None

            outputs = model(batch)
            recon, mu, logvar, z_recon = outputs

            if z_target is not None:
                loss = vae_loss_fn(recon, batch, mu, logvar, z_recon=z_recon, z_target=z_target)
            else:
                loss = F.mse_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    return model
