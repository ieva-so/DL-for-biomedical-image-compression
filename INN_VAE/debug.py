import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.inn import build_inn
from training.train_conv_vae import ConvVAE


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class BBBC005Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.image_paths = sorted(list(self.root.glob("*.tif")) + list(self.root.glob("*.TIF")))
        self.transform = transform or (lambda x: x)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = default_loader(self.image_paths[idx])
        return self.transform(img), 0

def get_dataloader(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset = BBBC005Dataset("./data/BBBC005_v1_images/subset", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def show_latent_maps(z, z_hat):
    z = z[0].cpu().numpy()
    z_hat = z_hat[0].cpu().numpy()
    error = np.abs(z - z_hat).mean(axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(z.transpose(1, 2, 0).clip(0, 1))
    ax[0].set_title("Original z")
    ax[1].imshow(z_hat.transpose(1, 2, 0).clip(0, 1))
    ax[1].set_title("Reconstructed ẑ")
    ax[2].imshow(error, cmap="hot")
    ax[2].set_title("Abs Error (z - ẑ)")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    dataloader = get_dataloader(batch_size=1)
    x, _ = next(iter(dataloader))
    x = x.to(DEVICE)

    # Load INN
    inn = build_inn(channels=3).to(DEVICE)
    inn.load_state_dict(torch.load("./training/checkpoints/inn_model.pth", map_location=DEVICE))
    inn.eval()

    # Load ConvVAE
    conv_vae = ConvVAE(in_channels=3, latent_dim=128).to(DEVICE)
    conv_vae.load_state_dict(torch.load("./training/checkpoints/conv_vae_model.pth", map_location=DEVICE))
    conv_vae.eval()

    with torch.no_grad():
        # 1. Check if INN works standalone (x -> z -> x̂)
        z = inn([x])[0]
        x_hat_inn = inn([z], rev=True)[0]
        inn_mse = F.mse_loss(x_hat_inn, x).item()
        print(f"[INN only] x → z → x̂ MSE: {inn_mse:.6f}")

        # 2. Check if ConvVAE preserves z (z -> ẑ)
        z_hat, _, _ = conv_vae(z)
        conv_vae_mse = F.mse_loss(z_hat, z).item()
        print(f"[ConvVAE] z → ẑ MSE: {conv_vae_mse:.6f}")

        # 3. Show latent recon quality
        show_latent_maps(z, z_hat)

        # 4. Reconstruct x̂ from ẑ
        x_hat_from_vae = inn([z_hat], rev=True)[0]
        final_mse = F.mse_loss(x_hat_from_vae, x).item()
        print(f"[Full] x → z → ẑ → x̂ MSE: {final_mse:.6f}")

        # 5. Save visual
        x_np = x[0].cpu().numpy().transpose(1, 2, 0)
        x_hat_np = x_hat_from_vae[0].cpu().numpy().transpose(1, 2, 0)
        diff = np.abs(x_np - x_hat_np).mean(axis=2)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(x_np.clip(0, 1))
        ax[0].set_title("Original x")
        ax[1].imshow(x_hat_np.clip(0, 1))
        ax[1].set_title("Final x̂")
        ax[2].imshow(diff, cmap="hot")
        ax[2].set_title("Abs Error")
        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.show()

        x_np = x[0].cpu().numpy().transpose(1, 2, 0)
        x_hat_inn_np = x_hat_inn[0].cpu().numpy().transpose(1, 2, 0)
        diff_inn = np.abs(x_np - x_hat_inn_np).mean(axis=2)

        x_hat_inn = inn([z], rev=True)[0]
        inn_mse = F.mse_loss(x_hat_inn, x).item()
        print(f"[INN only] x → z → x̂ MSE: {inn_mse:.6f}")
        x = x.to(DEVICE)
        z = inn([x])[0]
        x_hat = inn([z], rev=True)[0]

        def size_kb(tensor):
            return tensor.numel() * 4 / 1024  # float32 = 4 bytes

        print(f"Input x size:       {size_kb(x):.2f} KB")
        print(f"Latent z size:      {size_kb(z):.2f} KB")
        print(f"Reconstructed x̂ size: {size_kb(x_hat):.2f} KB")
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(x_np.clip(0, 1))
        ax[0].set_title("Original x")
        ax[1].imshow(x_hat_inn_np.clip(0, 1))
        ax[1].set_title("INN-Reconstructed x̂")
        ax[2].imshow(diff_inn, cmap="hot")
        ax[2].set_title("Abs Error (x - x̂)")
        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()