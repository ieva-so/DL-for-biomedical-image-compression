import torch
import torch.nn as nn
import torch.optim as optim
from nflows.flows.base import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.coupling import AffineCouplingTransform
import math

class MLPTransformNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x, context=None):
        return self.net(x)

def create_realnvp(input_dim, hidden_dim=256, num_blocks=6):
    transforms = []
    for i in range(num_blocks):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(
            AffineCouplingTransform(
                mask=torch.arange(input_dim) % 2,
                transform_net_create_fn=lambda in_features, out_features:
                    MLPTransformNet(in_features, out_features, hidden_dim)
            )
        )
    return Flow(CompositeTransform(transforms), StandardNormal([input_dim]))

def train_realnvp(model, dataloader, epochs=10, lr=1e-3, save_path="checkpoints/realnvp_model.pth"):
    model.to(next(model.parameters()).device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bpd = 0.0
        total_elements = 0

        for batch in dataloader:
            batch = batch.to(next(model.parameters()).device)
            batch_flat = batch.view(batch.size(0), -1)

            # Normalize latent vectors
            z_mean = batch_flat.mean(dim=1, keepdim=True)
            z_std = batch_flat.std(dim=1, keepdim=True) + 1e-6
            batch_flat = (batch_flat - z_mean) / z_std

            # Forward pass: log likelihood under flow model
            nll = -model.log_prob(inputs=batch_flat).mean()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += nll.item() * batch.size(0)
            total_elements += batch_flat.numel()
            total_bpd += nll.item() * batch.size(0) / (batch_flat.numel() * math.log(2))

        avg_loss = total_loss / len(dataloader.dataset)
        avg_bpd = total_bpd / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg NLL: {avg_loss:.3f}, Avg bits/dim: {avg_bpd:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved RealNVP model to {save_path}")