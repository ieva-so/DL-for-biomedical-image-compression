import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training.train_conv_vae import ConvVAE
from models.inn import build_inn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

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

# --------------------------
# Compression and Decompression
# --------------------------
def compress(x, inn_model, conv_vae):
    """
    Compresses an input image tensor `x` into latent encoding `mu`.
    Returns:
        - mu: compressed latent
        - z: INN-transformed representation of image
    """
    z = inn_model([x])[0]  # INN forward pass
    mu, _ = conv_vae.encode(z)  # VAE encoder
    return mu, z

def decompress(z_latent, conv_vae, inn_model):
    """
    Decompresses latent vector `z_latent` using ConvVAE decoder and INN reverse.
    Returns:
        - x_recon: reconstructed image
    """
    z_recon = conv_vae.decode(z_latent)
    x_recon = inn_model([z_recon], rev=True)[0]
    return x_recon

# --------------------------
# Evaluation and Visualization
# --------------------------
def decompress_and_compare():
    print("Loading INN and ConvVAE models...")
    inn_model = build_inn(channels=3).to(DEVICE)
    inn_model.load_state_dict(torch.load("../training/checkpoints/inn_model.pth", map_location=DEVICE))
    inn_model.eval()

    conv_vae = ConvVAE(in_channels=3, latent_dim=2048).to(DEVICE)
    conv_vae.load_state_dict(torch.load("../training/checkpoints/conv_vae_model_low_compression.pth", map_location=DEVICE))
    conv_vae.eval()

    print("Fetching sample image...")
    x, _ = get_sample()
    x = x.to(DEVICE)

    with torch.no_grad():
        # Compress
        z_latent, z = compress(x, inn_model, conv_vae)

        # Report compression stats
        element_size = 4  # float32
        BYTES_PER_KB = 1024
        z_kb = z[0].numel() * element_size / BYTES_PER_KB
        z_latent_kb = z_latent[0].numel() * element_size / BYTES_PER_KB
        print(f"Original z size:      {z_kb:.2f} KB")
        print(f"Latent vector size:   {z_latent_kb:.2f} KB")
        print(f"Compression Ratio:    {z_kb / z_latent_kb:.2f}x")

        # Decompress
        x_recon = decompress(z_latent, conv_vae, inn_model)

        # MSE
        mse = F.mse_loss(x, x_recon).item()
        print(f"Reconstruction MSE:   {mse:.6f}")

        # Visualization
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
        Path("outputs").mkdir(exist_ok=True)
        plt.savefig("outputs/decompressed_comparison_low_compression.png")
        plt.close()
        print("Saved visualization to outputs/decompressed_comparison_low_compression.png")

if __name__ == "__main__":
    decompress_and_compare()
