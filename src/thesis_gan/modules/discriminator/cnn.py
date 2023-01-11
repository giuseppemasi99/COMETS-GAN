from math import factorial

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from thesis_gan.common.utils import corr


def linear_block(in_features: int, out_features: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        spectral_norm(nn.Linear(in_features, out_features), n_power_iterations=10),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


def conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        spectral_norm(nn.Conv1d(in_channels, out_channels, (3,), padding="same"), n_power_iterations=10),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


class CNNDiscriminator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim: int,
        compute_corr: bool,
        alpha: float,
    ) -> None:
        super(CNNDiscriminator, self).__init__()
        self.compute_corr = compute_corr
        self.alpha = alpha
        self.n_features = n_features

        self.convblock1 = nn.Sequential(conv_block(n_features, 16, dropout), nn.MaxPool1d(2))
        self.convblock2 = nn.Sequential(conv_block(16, 16, dropout), nn.MaxPool1d(2))
        self.convblock3 = conv_block(16, 16, dropout)
        self.linear1 = linear_block((encoder_length + decoder_length) // 4 * 16, hidden_dim * 4, dropout)
        self.linear2 = linear_block(hidden_dim * 4, hidden_dim * 2, dropout)
        self.linear3 = linear_block(hidden_dim * 2, hidden_dim * 1, dropout)
        self.linear_out = spectral_norm(nn.Linear(hidden_dim, 1), n_power_iterations=10)

        if compute_corr and n_features > 1:
            corr_features = factorial(n_features) // (2 * factorial(n_features - 2))
            self.linear_corr = spectral_norm(nn.Linear(corr_features, 1), n_power_iterations=10)

    def forward(self, x: torch.Tensor, y_continuation: torch.Tensor) -> torch.Tensor:

        concatenated = torch.cat((x, y_continuation), dim=-1)

        o = self.convblock1(concatenated)
        o = self.convblock2(o)
        o = self.convblock3(o).flatten(start_dim=1)
        o = self.linear1(o)
        o = self.linear2(o)
        o = self.linear3(o)

        output = self.linear_out(o)

        if self.n_features > 1 and self.compute_corr:
            correlations = corr(y_continuation)
            corr_score = self.linear_corr(correlations)
            if self.alpha is not None:
                output = self.alpha * output + (1 - self.alpha) * corr_score
            else:
                output += corr_score

        return output
