import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.inn import build_inn
import os
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
from pathlib import Path

# Use MPS (Metal Performance Shaders) if available on Mac
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class BBBC005Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.image_paths = sorted(list(self.root.glob("*.tif")) + list(self.root.glob("*.TIF")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No .tif or .TIF images found in {self.root.resolve()}")
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = default_loader(self.image_paths[idx])
        return self.transform(image), 0  # dummy label

def get_dataloader(batch_size=16):
    from torchvision.datasets.folder import default_loader
    from pathlib import Path

    class BBBC005Dataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = Path(root)
            self.image_paths = sorted(list(self.root.glob("*.tif")) + list(self.root.glob("*.TIF")))
            if len(self.image_paths) == 0:
                raise FileNotFoundError(f"No .tif or .TIF images found in {self.root.resolve()}")
            self.transform = transform or (lambda x: x)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = default_loader(self.image_paths[idx])
            return self.transform(image), 0  # dummy label

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    dataset = BBBC005Dataset(root="../data/BBBC005_v1_images/subset", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def visualize_reconstruction(x, x_recon, epoch, batch_idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy(), cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(x_recon[0].detach().permute(1, 2, 0).cpu().numpy(), cmap='gray')
    axes[1].set_title('Reconstructed')
    error = (x[0] - x_recon[0]).detach().abs().permute(1, 2, 0).cpu().numpy()
    axes[2].imshow(error.clip(0, 1), cmap='hot')
    axes[2].set_title('Abs Error')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch+1}, Batch {batch_idx}")
    plt.tight_layout()
    plt.savefig(f"recon_vis_epoch{epoch+1}_batch{batch_idx}.png")
    plt.close()

def train_inn_model(epochs=20, batch_size=8, lr=1e-4, save_path="checkpoints/inn_model.pth"):
    dataloader = get_dataloader(batch_size)
    model = build_inn(channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(DEVICE)
            z = model([x])[0]  # Forward pass: wrap input in list and unpack output
            # Inject Gaussian noise to simulate ConvVAE-like distortion
            noise_std = torch.empty_like(z).uniform_(0.01, 0.1)
            z_noisy = z + noise_std * torch.randn_like(z)
            # Train the INN to reconstruct from slightly perturbed latent
            x_recon = model([z_noisy], rev=True)[0]

            recon_loss = nn.MSELoss()(x_recon, x)
            latent_penalty = z.pow(2).mean()
            loss = recon_loss + 1e-4 * latent_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # --- Compression Estimation ---
            original_bits = x.numel() * 32  # float32 input
            compressed_bits = z.numel() * 8  # pretend we use 8-bit entropy-coded z
            compression_ratio = original_bits / compressed_bits

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.6f}, z mean: {z.mean().item():.4f}, z std: {z.std().item():.4f}")
                print(f"Compression Ratio Estimate: {compression_ratio:.2f}x")
                visualize_reconstruction(x, x_recon, epoch, batch_idx)

        print(f"Epoch [{epoch+1}/{epochs}] finished. Avg Loss: {total_loss/len(dataloader):.6f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_inn_model()
