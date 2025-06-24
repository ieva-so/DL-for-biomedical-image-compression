import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.train_vae import vae_loss_fn
from training.train_conv_vae import ConvVAE
from models.inn import build_inn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path
import matplotlib.pyplot as plt
import random

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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
        return self.transform(image), 0

def get_dataloader(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset = BBBC005Dataset(root="../data/BBBC005_v1_images/subset", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_compression():
    print("Loading trained INN and VAE models...")
    inn_model = build_inn(channels=3).to(DEVICE)
    inn_model.load_state_dict(torch.load("../training/checkpoints/inn_model.pth", map_location=DEVICE))

    dataloader = get_dataloader()

    sample = next(iter(dataloader))[0][:50].to(DEVICE)  # get 50 samples
    with torch.no_grad():
        z = inn_model([sample])[0]
        vae = ConvVAE(in_channels=z.size(1), latent_dim=128).to(DEVICE)
        vae.load_state_dict(torch.load("../training/checkpoints/conv_vae_model.pth", map_location=DEVICE))
        vae.eval()

        z_recon, _, _ = vae(z)
        x_recon = inn_model([z_recon], rev=True)[0]

    print("Generating visual comparisons for 50 samples...")
    output_dir = Path("outputs/image_vs_recon")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(sample.size(0)):
        x_img = sample[i].cpu().numpy()
        x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min() + 1e-8)

        x_hat = x_recon[i].detach().cpu().numpy()
        x_hat = (x_hat - x_hat.min()) / (x_hat.max() - x_hat.min() + 1e-8)

        mse = F.mse_loss(torch.tensor(x_img), torch.tensor(x_hat)).item()
        print(f"Sample {i+1:02d} MSE: {mse:.6f}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x_img.transpose(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[1].imshow(x_hat.transpose(1, 2, 0))
        axes[1].set_title('Reconstructed Image')
        diff = abs(x_img - x_hat).mean(axis=0)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Abs Error')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"reconstructed_image_{i+1:02d}.png")
        plt.close()

    print(f"Saved 50 reconstructed image comparisons to {output_dir}/")

if __name__ == "__main__":
    evaluate_compression()