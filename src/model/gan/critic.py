from math import factorial

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class CNNCritic(nn.Module):

    def __init__(
        self, encoder_length: int, decoder_length: int, n_features: int,
        dropout: float, hidden_dim: int, # ckpt_path: str
    ) -> None:
        super(CNNCritic, self).__init__()
        self.n_features = n_features

        self.convblock1 = nn.Sequential(self.conv_block(n_features, 16, dropout), nn.MaxPool1d(2))
        self.convblock2 = nn.Sequential(self.conv_block(16, 16, dropout), nn.MaxPool1d(2))
        self.convblock3 = self.conv_block(16, 16, dropout)
        self.linear1 = self.linear_block((encoder_length + decoder_length) // 4 * 16, hidden_dim * 4, dropout)
        self.linear2 = self.linear_block(hidden_dim * 4, hidden_dim * 2, dropout)
        self.linear3 = self.linear_block(hidden_dim * 2, hidden_dim * 1, dropout)
        self.linear_out = spectral_norm(nn.Linear(hidden_dim, 1), n_power_iterations=10)

        if n_features > 1:
            corr_features = factorial(n_features) // (2 * factorial(n_features - 2))
            self.linear_corr = spectral_norm(nn.Linear(corr_features, 1), n_power_iterations=10)

        # self.load_state_dict(torch.load(ckpt_path))

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t.shape = [batch_size, seq_len, n_features]
        x_t = x_t.transpose(2, 1)

        # o = self.convblock1(x_t)
        # o = self.convblock2(o)
        # o = self.convblock3(o).flatten(start_dim=1)
        # o = self.linear1(o)
        # o = self.linear2(o)
        # o = self.linear3(o)

        # output = self.linear_out(o)

        # if self.n_features > 1:
        correlations = self.corr(x_t)
        score_corr = self.linear_corr(correlations)
        
        # output += score_corr

        return score_corr
    
    def corr(self, x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch.shape = [batch_size, n_features, decoder_steps]

        indices = torch.triu_indices(self.n_features, self.n_features, 1)

        correlations = []
        for x in x_batch:
            # x.shape = [n_features, decoder_steps]
            correlation = torch.corrcoef(x)
            # correlation.shape = [n_features, n_features]
            correlation = correlation[indices[0], indices[1]]
            # correlation.shape = [bin(n_features, 2)]

            correlations.append(torch.nan_to_num(correlation))

        correlations = torch.stack(correlations)
        # correlations.shape = [batch_size, bin(n_features, 2)]

        return correlations

    @staticmethod
    def linear_block(in_features: int, out_features: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features), n_power_iterations=10),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

    @staticmethod
    def conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, out_channels, (3,), padding="same"), n_power_iterations=10),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )