import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.inn import build_inn
from utils.latent_dataset import LatentDataset
from train_diffusion import train_diffusion
from utils.estimate_entropy import estimate_entropy
from torchvision.datasets.folder import default_loader
from pathlib import Path
from train_vae import VAE, train_vae
from train_conv_vae import ConvVAE
from train_real_nvp import create_realnvp, train_realnvp
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load and preprocess dataset
def get_dataloader(batch_size=16):

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


if __name__ == "__main__":
    print("Loading INN model...")
    inn_model = build_inn(channels=3).to(DEVICE)
    inn_model.load_state_dict(torch.load("checkpoints/inn_model.pth", map_location=DEVICE))

    print("Preparing data loader and latent dataset...")
    dataloader = get_dataloader()
    latent_dataset = LatentDataset(inn_model, dataloader, device=DEVICE)
    latent_loader = DataLoader(latent_dataset, batch_size=16, shuffle=True)

    print("Training ConvVAE over latent space...")
    conv_vae = ConvVAE(in_channels=latent_dataset[0].size(0), latent_dim=128)
    train_vae(conv_vae, latent_loader, epochs=250, device=DEVICE)
    torch.save(conv_vae.state_dict(), "checkpoints/conv_vae_model.pth")
    print("Saved ConvVAE model to checkpoints/conv_vae_model.pth")

    # print("Training original VAE over flattened latent space...")
    # input_dim = latent_dataset[0].numel()
    # vae = VAE(input_dim=input_dim, latent_dim=32)
    # train_vae(vae, latent_loader, epochs=10, device=DEVICE)
    # torch.save(vae.state_dict(), "checkpoints/vae_model.pth")
    # print("Saved VAE model to checkpoints/vae_model.pth")

    # print("Training RealNVP over latent space...")
    # input_dim = latent_dataset[0].numel()  # flatten z from (C,H,W) → (C*H*W)
    # realnvp = create_realnvp(input_dim)
    # train_realnvp(realnvp, latent_loader, epochs=50, save_path="checkpoints/realnvp_model.pth")
    # print("Saved RealNVP model to checkpoints/realnvp_model.pth")

    print("Pipeline complete.")
