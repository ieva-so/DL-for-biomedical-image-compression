import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.latent_dataset import LatentDataset

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.fc_decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss_fn(recon_x, x, mu, logvar, z_recon=None, z_target=None, beta=0.01, lambda_z=0.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    z_loss = F.mse_loss(z_recon, z_target, reduction='sum') if z_recon is not None and z_target is not None else 0.0
    return recon_loss + beta * kld + lambda_z * z_loss

def train_vae(model, dataloader, epochs=100, lr=1e-3, device="cpu", save_path="checkpoints/vae_model.pth", z_targets=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            z_target = z_targets[i * batch.size(0):(i + 1) * batch.size(0)].to(device) if z_targets is not None else None

            outputs = model(batch)
            recon, mu, logvar, z_recon = outputs

            loss = vae_loss_fn(recon, batch, mu, logvar, z_recon=z_recon, z_target=z_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bpd = loss.item() / batch.numel() / torch.log(torch.tensor(2.0))
            print(f"\tBatch BPD: {bpd:.4f}")

        avg_bpd = total_loss / (len(dataloader.dataset) * dataloader.dataset[0].numel()) / torch.log(torch.tensor(2.0))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}, Avg BPD: {avg_bpd:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return model
