from math import factorial
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        use_norm: bool = True,
    ) -> None:
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, (kernel_size,), stride=(stride,), padding=padding, dilation=(dilation,)
        )
        if use_norm:
            self.conv1 = spectral_norm(self.conv1, n_power_iterations=10)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, (kernel_size,), stride=(stride,), padding=padding, dilation=(dilation,)
        )
        if use_norm:
            self.conv2 = spectral_norm(self.conv2, n_power_iterations=10)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, (1,)) if n_inputs != n_outputs else None
        if n_inputs != n_outputs and use_norm:
            self.downsample = spectral_norm(self.downsample, n_power_iterations=10)

        self.relu = nn.LeakyReLU(0.2)
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class NoisyTemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super(NoisyTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1] + 1
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    use_norm=False,
                )
            ]

        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = torch.cat((x, noise), dim=1)
            x = layer(x)
        return x


def corr(x_batch: torch.Tensor) -> torch.Tensor:
    n_features = x_batch.shape[1]
    indices = torch.triu_indices(n_features, n_features, 1)

    correlations = []
    for x in x_batch:
        correlation = torch.corrcoef(x)
        correlation = correlation[indices[0], indices[1]]
        correlations.append(correlation)
    return torch.stack(correlations)


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

        self.tcn = NoisyTemporalConvNet(n_features + 1, [32, 64, 128, 64, 32, 16, n_features], dropout=dropout)
        self.linear_out = nn.Linear(encoder_length, decoder_length)

    def forward(self, batch: Dict[str, torch.Tensor], noise: torch.Tensor):
        x = batch["x"]
        o = self.tcn(x, noise)
        return self.linear_out(o)


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim: int,
        compute_corr: bool,
    ) -> None:
        super(ConditionalDiscriminator, self).__init__()
        self.compute_corr = compute_corr
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

        if self.n_features > 1 and self.compute_corr:
            correlations = corr(y_continuation)
            output += self.linear_corr(correlations)

        return output
