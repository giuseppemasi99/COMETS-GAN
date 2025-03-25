import math

import torch
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    B, T = timesteps.shape
    timesteps = timesteps.flatten()

    half_dim: int = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb.reshape(B, T, embedding_dim)



