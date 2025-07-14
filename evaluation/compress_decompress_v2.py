import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path
from PIL import Image
import random
import sys
import os

current_dir = os.path.dirname(__file__)
inn_vae_path = os.path.abspath(os.path.join(current_dir, '..', 'INN_VAE'))
sys.path.append(inn_vae_path)
from models.inn import build_inn
from training.train_conv_vae import ConvVAE


DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print("Using device:", DEVICE)

# Dataset
class BBBC005Dataset(torch.utils.data.Dataset):
    def _init_(self, root, transform=None):
        self.root = Path(root)
        self.image_paths = sorted(list(self.root.glob(".tif")) + list(self.root.glob(".TIF")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.root.resolve()}")
        self.transform = transform or (lambda x: x)

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        image = default_loader(self.image_paths[idx])
        return self.transform(image), 0

def get_sample(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = BBBC005Dataset(
        root="C:\\Users\\ievas\\Desktop\\UNI\\00 SEMESTERS\\SoSe25\\Project Seminar Biomedical Image Analysis\\data\\dataset\\rawimages",
        transform=transform
    )
    indices = random.sample(range(len(dataset)), batch_size)
    samples = [dataset[i][0].unsqueeze(0) for i in indices]
    return torch.cat(samples, dim=0), 0  # dummy label

def load_single_image(image_path, image_size=(256,256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dim
    return tensor.to(DEVICE)

# --------------------------
# Compression and Decompression
# --------------------------
def compress(x, inn_model, conv_vae):
    z = inn_model([x])[0]  # INN forward
    mu, _ = conv_vae.encode(z)
    return mu, z

def decompress(z_latent, conv_vae, inn_model):
    z_recon = conv_vae.decode(z_latent)
    x_recon = inn_model([z_recon], rev=True)[0]
    return x_recon

# --------------------------
# Evaluation and Visualization
# --------------------------
def decompress_and_compare():
    print("Loading models...")
    inn_model = build_inn(channels=3, height=64, width=64).to(DEVICE)
    inn_model.load_state_dict(torch.load("C:\\Users\\ievas\\Desktop\\UNI\\00 SEMESTERS\\SoSe25\\Project Seminar Biomedical Image Analysis\\data\\checkpoints\\inn_model.pth", map_location=DEVICE))
    inn_model.eval()

    # 🔁 Infer channel count for ConvVAE from INN output
    dummy_input = torch.randn(1, 3, 64, 64).to(DEVICE)
    z_dummy = inn_model([dummy_input])[0]
    in_channels = z_dummy.shape[1]

    conv_vae = ConvVAE(in_channels=in_channels, latent_dim=512).to(DEVICE)
    conv_vae.load_state_dict(torch.load("C:\\Users\\ievas\\Desktop\\UNI\\00 SEMESTERS\\SoSe25\\Project Seminar Biomedical Image Analysis\\data\\checkpoints\\conv_vae_model_low_compression.pth", map_location=DEVICE))
    conv_vae.eval()

    print("Fetching sample image...")
    x, _ = get_sample()
    x = x.to(DEVICE)

    with torch.no_grad():
        # Compress
        z_latent, z = compress(x, inn_model, conv_vae)

        # Compression Stats
        element_size = 4  # float32
        z_kb = z[0].numel() * element_size / 1024
        z_latent_kb = z_latent[0].numel() * element_size / 1024
        print(f"Original z size:    {z_kb:.2f} KB")
        print(f"Latent vector size: {z_latent_kb:.2f} KB")
        print(f"Compression Ratio:  {z_kb / z_latent_kb:.2f}x")

        # Decompress
        x_recon = decompress(z_latent, conv_vae, inn_model)

        # MSE
        mse = F.mse_loss(x, x_recon).item()
        print(f"Reconstruction MSE: {mse:.6f}")

        # Visualization
        x_np = x[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        xhat_np = x_recon[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        diff = abs(x_np - xhat_np).mean(axis=2)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_np)
        axes[0].set_title("Original")
        axes[1].imshow(xhat_np)
        axes[1].set_title("Reconstructed")
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title("Abs Error")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        Path("/content/drive/MyDrive/outputs").mkdir(exist_ok=True)
        plt.savefig("/content/drive/MyDrive/outputs/decompressed_comparison_low_compression.png")
        plt.close()
        print("Saved: decompressed_comparison_low_compression.png")

    if __name__ == "__main__":
        decompress_and_compare()