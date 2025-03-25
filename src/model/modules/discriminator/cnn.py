from math import factorial

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from common.utils import corr
from model.modules.non_linearity import non_linearity
from model.modules.timestep_embedding import get_timestep_embedding


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
        self, encoder_length: int, decoder_length: int,
        hidden_dim: int, dropout: float, n_features, 
        t_embedding_dim: int = 128
    ) -> None:
        super(CNNDiscriminator, self).__init__()
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_features = n_features
        self.n_stocks = n_features // 2
        self.t_embedding_dim = t_embedding_dim

        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(t_embedding_dim, t_embedding_dim*2), nn.Linear(t_embedding_dim*2, 1)])

        self.convblock1 = nn.Sequential(conv_block(n_features+1, 16, dropout), nn.MaxPool1d(2))
        self.convblock2 = nn.Sequential(conv_block(16, 16, dropout), nn.MaxPool1d(2))
        self.convblock3 = conv_block(16, 16, dropout)
        self.linear1 = linear_block((encoder_length + decoder_length) // 4 * 16, hidden_dim * 4, dropout)
        self.linear2 = linear_block(hidden_dim * 4, hidden_dim * 2, dropout)
        self.linear3 = linear_block(hidden_dim * 2, hidden_dim * 1, dropout)
        self.linear_out = spectral_norm(nn.Linear(hidden_dim, 1), n_power_iterations=10)

        corr_features = factorial(self.n_stocks) // (2 * factorial(self.n_stocks - 2))
        self.linear_corr = spectral_norm(nn.Linear(corr_features, 1), n_power_iterations=10)

    def forward(
        self, x: torch.Tensor, y_continuation: torch.Tensor, 
        t_past: torch.Tensor, t_fut: torch.Tensor
    ) -> torch.Tensor:
        # t_past.shape = [batch_size, encoder_length]
        # t_fut.shape = [batch_size, decoder_length]

        t_emb = torch.cat((t_past, t_fut), dim=-1)  # .unsqueeze(1)
        t_emb = get_timestep_embedding(t_emb, self.t_embedding_dim)
        t_emb = self.temb.dense[0](t_emb)
        t_emb = non_linearity(t_emb)
        t_emb = self.temb.dense[1](t_emb).transpose(2, 1)

        concatenated = torch.cat((x, y_continuation), dim=-1)
        concatenated = torch.cat((concatenated, t_emb), dim=1)

        o = self.convblock1(concatenated)
        o = self.convblock2(o)
        o = self.convblock3(o).flatten(start_dim=1)
        o = self.linear1(o)
        o = self.linear2(o)
        o = self.linear3(o)

        output = self.linear_out(o)
        output += self.linear_corr(corr(y_continuation[:, :self.n_stocks]))

        return output
