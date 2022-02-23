from typing import Dict

import torch
import torch.nn as nn


def linear_block(in_features: int, out_features: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


def conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, (3,), padding="same"),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


class CNNGenerator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        super(CNNGenerator, self).__init__()

        self.linearblock1 = linear_block(encoder_length, hidden_dim * 4, dropout)
        self.convblock1 = conv_block(n_features + 1, 16, dropout)
        self.convblock2 = conv_block(16, 16, dropout)
        self.convblock3 = conv_block(16, n_features, dropout)

        self.linear_out = nn.Linear(hidden_dim * 4, decoder_length)

    def forward(self, batch: Dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
        x = batch["x"]
        x_noise = torch.cat((x, noise), dim=1)

        o = self.linearblock1(x_noise)
        o = self.convblock1(o)
        o = self.convblock2(o)
        o = self.convblock3(o)

        output = self.linear_out(o)
        return output
