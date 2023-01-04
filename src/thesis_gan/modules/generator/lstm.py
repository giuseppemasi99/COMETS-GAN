import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        n_stocks: int,
        is_volumes: int,
        dropout: float,
        hidden_dim=None,
    ) -> None:
        super(LSTMGenerator, self).__init__()
        self.n_stocks = n_stocks
        self.is_volumes = is_volumes

        self.linear1 = nn.Linear(n_features + 1, n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=encoder_length,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
            proj_size=n_features,
        )

        self.linear2 = nn.Linear(encoder_length, decoder_length)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # noise.shape = [batch_size, 1, encoder_length]
        # x.shape = [batch_size, n_features, encoder_length]

        x = torch.cat((x, noise), dim=1)
        # x.shape = [batch_size, n_features+1, encoder_length]

        x = torch.permute(x, (0, 2, 1))
        # x.shape = [batch_size, encoder_length, n_features+1]

        o = self.linear1(x)
        # o.shape = [batch_size, encoder_length, n_features]

        o, _ = self.lstm(o)
        o = torch.permute(o, (0, 2, 1))
        o = self.linear(o)
        # o.shape = [batch_size, n_features, decoder_length]

        if self.is_volumes:
            o_price, o_volume = o[:, : self.n_stocks, :], o[:, self.n_stocks :, :]

            # o_volume = torch.abs(o_volume)
            o_volume = self.tanh(o_volume)

            o = torch.concatenate((o_price, o_volume), dim=1)

        # o.shape = [batch_size, n_features, decoder_length]
        return o
