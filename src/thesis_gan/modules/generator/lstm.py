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
        hidden_dim1=256,
        hidden_dim2=192,
    ) -> None:
        super(LSTMGenerator, self).__init__()
        self.n_stocks = n_stocks
        self.is_volumes = is_volumes

        self.linear1 = nn.Linear(n_features + 1, n_features)

        self.lstm1 = nn.LSTM(
            input_size=n_features + 1,
            hidden_size=hidden_dim1,
            batch_first=True,
            dropout=dropout,
        )

        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=decoder_length,
            batch_first=True,
            dropout=dropout,
            proj_size=n_features,
        )

        self.linear3 = nn.Linear(encoder_length, decoder_length)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, noise: torch.Tensor):

        x = torch.cat((x, noise), dim=1)

        x = torch.permute(x, (0, 2, 1))

        o, _ = self.lstm1(x)

        o = self.linear2(o)

        o = self.dropout(self.relu(o))

        o, _ = self.lstm2(o)

        o = torch.permute(o, (0, 2, 1))

        o = self.linear3(o)

        if self.is_volumes:
            o_price, o_volume = o[:, : self.n_stocks, :], o[:, self.n_stocks :, :]

            # o_volume = torch.abs(o_volume)
            o_volume = self.tanh(o_volume)

            o = torch.concatenate((o_price, o_volume), dim=1)

        return o
