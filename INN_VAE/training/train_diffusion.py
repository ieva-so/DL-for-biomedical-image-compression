import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
import torch.optim as optim
from utils.latent_dataset import LatentDataset

def train_diffusion(model_inn, dataloader, device="cpu", epochs=10, batch_size=16, lr=1e-4):
    latent_dataset = LatentDataset(model_inn, dataloader, device)
    latent_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True)

    z_shape = latent_dataset[0].shape
    model = UNet2DModel(
    sample_size=z_shape[-1],
    in_channels=z_shape[0],
    out_channels=z_shape[0],
    layers_per_block=1,
    block_out_channels=(64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
    add_attention=False,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("epoch start: ", epoch)
        count = 0
        for batch in latent_loader:
            count = count + 1
            print("epoch inside: ", count)
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch.size(0),), device=device).long()
            noisy_latents = scheduler.add_noise(batch, noise, timesteps)

            noise_pred = model(noisy_latents, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "checkpoints/diffusion_model.pth")
    return model, scheduler
