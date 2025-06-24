import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path
from models.inn import build_inn
from training.train_real_nvp import create_realnvp

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class BBBC005Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.image_paths = sorted(list(self.root.glob("*.tif")) + list(self.root.glob("*.TIF")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.root.resolve()}")
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = default_loader(self.image_paths[idx])
        return self.transform(image), 0

def get_sample(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset = BBBC005Dataset(root="../data/BBBC005_v1_images/subset", transform=transform)
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size)))

def decompress_and_compare():
    print("Loading INN and RealNVP...")
    inn_model = build_inn(channels=3).to(DEVICE)
    inn_model.load_state_dict(torch.load("../training/checkpoints/inn_model.pth", map_location=DEVICE))
    inn_model.eval()

    print("Fetching sample image...")
    x, _ = get_sample()
    x = x.to(DEVICE)

    with torch.no_grad():
        # Encode image to latent z
        z = inn_model([x])[0]
        z_flat = z.view(z.size(0), -1)

        # Load trained RealNVP
        realnvp = create_realnvp(input_dim=z_flat.size(1))
        realnvp = realnvp.float().to(DEVICE)
        realnvp.load_state_dict(torch.load("../training/checkpoints/realnvp_model.pth", map_location=DEVICE))
        realnvp.eval()

        # Compress: encode z -> noise via RealNVP
        noise, _ = realnvp._transform(z_flat)

        # Estimate size
        z_size_kb = z_flat[0].numel() * 4 / 1024
        noise_size_kb = noise[0].numel() * 4 / 1024
        print(f"Original z size: {z_size_kb:.2f} KB")
        print(f"Encoded latent size: {noise_size_kb:.2f} KB")
        print(f"Compression Ratio: {z_size_kb / noise_size_kb:.2f}x")

        # Decode noise -> ẑ
        z_recon, _ = realnvp.inverse(noise)
        z_recon = z_recon.view_as(z)
        x_recon = inn_model([z_recon], rev=True)[0]

    mse = F.mse_loss(x, x_recon).item()
    print(f"Reconstruction MSE: {mse:.6f}")

    x_np = x[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    xhat_np = x_recon[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    diff = abs(x_np - xhat_np).mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_np)
    axes[0].set_title("Original Image")
    axes[1].imshow(xhat_np)
    axes[1].set_title("Reconstructed Image")
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Abs Error")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/realnvp_comparison.png")
    print("Saved visualization to outputs/realnvp_comparison.png")

if __name__ == "__main__":
    decompress_and_compare()