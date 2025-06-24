import torch
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, model, dataloader, device="cpu"):
        self.latents = []
        model.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                z = model([x])[0].detach().cpu()
                self.latents.append(z)
        self.latents = torch.cat(self.latents, dim=0)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]
