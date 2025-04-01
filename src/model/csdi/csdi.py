import math

import torch
import torch.nn as nn

from model.csdi.conv1d_with_init import Conv1dWithInit
from model.csdi.diffusion_embedding import DiffusionEmbedding
from model.csdi.residual_block import ResidualBlock


class CSDI(nn.Module):
    def __init__(self, channels, diffusion_timesteps, diffusion_embedding_dim, nheads, is_linear, layers):
        super().__init__()
        self.input_dim = 1
        self.channels = channels

        self.diffusion_embedding = DiffusionEmbedding(
            diffusion_timesteps=diffusion_timesteps,
            diffusion_embedding_dim=diffusion_embedding_dim,
        )

        self.input_projection = Conv1dWithInit(self.input_dim, self.channels)
        self.output_projection1 = Conv1dWithInit(self.channels, self.channels)
        self.output_projection2 = Conv1dWithInit(self.channels, 1)
        nn.init.zeros_(self.output_projection2.layer.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    nheads=nheads,
                    is_linear=is_linear,
                )
                for _ in range(layers)
            ]
        )

        self.relu = nn.ReLU()

    def forward(self, x, diffusion_step):
        B, K, L = x.shape
        x = x.reshape(B, self.input_dim, K * L)
        x = self.input_projection(x)
        x = self.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)   
        x = self.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        
        return x
