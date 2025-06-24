import torch
from diffusers import UNet2DModel, DDPMScheduler

def estimate_entropy(model, scheduler, z_loader, device="mps"):
    model.eval()
    neg_log_likelihood_total = 0.0
    count = 0

    with torch.no_grad():
        for z in z_loader:
            z = z.to(device)
            # Placeholder likelihood estimator — actual would use ELBO or diffusion likelihood estimator
            # Here we sum MSE between z and denoised prediction as a proxy
            noise = torch.randn_like(z)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (z.size(0),), device=device).long()
            noisy_latents = scheduler.add_noise(z, noise, timesteps)

            noise_pred = model(noisy_latents, timesteps).sample
            neg_log_likelihood = torch.nn.functional.mse_loss(noise_pred, noise, reduction='sum').item()

            neg_log_likelihood_total += neg_log_likelihood
            count += z.numel()

    bits = neg_log_likelihood_total / torch.log(torch.tensor(2.0)).item()
    bpp = bits / count
    print(f"Estimated bits: {bits:.2f}, bits per latent dimension: {bpp:.6f}")
    return bits, bpp
