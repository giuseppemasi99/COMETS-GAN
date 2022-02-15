from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def linear_block(in_features: int, out_features: int, dropout: float, normalization: bool = True) -> nn.Module:
    if normalization:
        return nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features), n_power_iterations=10),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )


def conv_block(in_channels: int, out_channels: int, dropout: float, normalization: bool = True) -> nn.Module:
    if normalization:
        return nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, out_channels, (3,), padding="same"), n_power_iterations=10),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, (3,), padding="same"),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        super(ConditionalGenerator, self).__init__()

        self.linearblock1 = linear_block(encoder_length, hidden_dim * 4, dropout, normalization=False)
        self.convblock1 = conv_block(n_features + 1, 16, dropout, normalization=False)
        self.convblock2 = conv_block(16, 16, dropout, normalization=False)
        self.convblock3 = conv_block(16, n_features, dropout, normalization=False)

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


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        super(ConditionalDiscriminator, self).__init__()

        self.flatten = nn.Flatten()
        self.convblock1 = nn.Sequential(conv_block(n_features, 16, dropout), nn.MaxPool1d(2))
        self.convblock2 = nn.Sequential(conv_block(16, 16, dropout), nn.MaxPool1d(2))
        self.convblock3 = conv_block(16, 16, dropout)
        self.linear1 = linear_block((encoder_length + decoder_length) // 4 * 16, hidden_dim * 4, dropout)
        self.linear2 = linear_block(hidden_dim * 4, hidden_dim * 2, dropout)
        self.linear3 = linear_block(hidden_dim * 2, hidden_dim * 1, dropout)
        self.linear_out = spectral_norm(nn.Linear(hidden_dim, 1), n_power_iterations=10)

    def forward(self, batch: Dict[str, torch.Tensor], y_continuation: torch.Tensor) -> torch.Tensor:
        x = batch["x"]

        concatenated = torch.cat((x, y_continuation), dim=-1)

        o = self.convblock1(concatenated)
        o = self.convblock2(o)
        o = self.convblock3(o).flatten(start_dim=1)
        o = self.linear1(o)
        o = self.linear2(o)
        o = self.linear3(o)

        output = self.linear_out(o)
        return output
