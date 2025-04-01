import torch
import torch.nn as nn


class DiffusionEmbedding(nn.Module):
    def __init__(self, diffusion_timesteps, diffusion_embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = diffusion_embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(diffusion_timesteps, diffusion_embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(diffusion_embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)
        self.silu = nn.SiLU()

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = self.silu(x)
        x = self.projection2(x)
        x = self.silu(x)
        return x

    def _build_embedding(self, diffusion_timesteps, dim=64):
        steps = torch.arange(diffusion_timesteps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table