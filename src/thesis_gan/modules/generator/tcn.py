from typing import List

import torch
import torch.nn as nn


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
    ) -> None:
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, (kernel_size,), stride=(stride,), padding=padding, dilation=(dilation,)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, (kernel_size,), stride=(stride,), padding=padding, dilation=(dilation,)
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, (1,)) if n_inputs != n_outputs else None

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
                )
            ]

        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = torch.cat((x, noise), dim=1)
            x = layer(x)
        return x


class TCNGenerator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        dropout: float,
        hidden_dim=None,
    ) -> None:
        super(TCNGenerator, self).__init__()
        self.n_stocks = int(n_features / 2)

        self.tcn = NoisyTemporalConvNet(n_features + 1, [32, 64, 128, 64, 32, 16, n_features], dropout=dropout)
        self.linear_out = nn.Linear(encoder_length, decoder_length)

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # print(x.shape)
        # x.shape = [batch_size, num_features, encoder_length]

        o = self.tcn(x, noise)
        o = self.linear_out(o)
        # o.shape = [batch_size, num_features, decoder_length]

        o_price, o_volume = o[:, : self.n_stocks, :], o[:, self.n_stocks :, :]
        o_volume = torch.abs(o_volume)

        o = torch.concatenate((o_price, o_volume), dim=1)
        # o.shape = [batch_size, num_features, decoder_length]

        return o
