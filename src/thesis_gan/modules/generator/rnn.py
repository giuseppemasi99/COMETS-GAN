import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(
        self,
        n_features,
        encoder_length,
        dropout,
        hidden_dim,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        self.n_features = n_features
        self.encoder_length = encoder_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            num_layers=num_layers,
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, x):
        # x.shape = [batch_size, encoder_length, n_features]
        gru_out, hidden = self.gru(x)

        if self.rnn_directions * self.num_layers > 1:
            if self.rnn_directions > 1:
                gru_out = gru_out.view(x.size(0), self.sequence_len, self.rnn_directions, self.hidden_dim)
                gru_out = torch.sum(gru_out, dim=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, x.size(0), self.hidden_dim)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(dim=0)
        else:
            hidden.squeeze_(0)

        return gru_out, hidden


class RNNDecoder(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_dim,
        dropout,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        self.n_features = n_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.out = nn.Linear(hidden_dim, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_hidden):
        o, h = self.gru(x, prev_hidden)
        o = self.out(o)
        return o, self.dropout(h)


# TODO: handle bidirectional RNNs
class RNNGenerator(nn.Module):
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        n_stocks: int,
        is_volumes: int,
        dropout: float,
        hidden_dim=None,
        num_layers=1,
    ) -> None:
        super(RNNGenerator, self).__init__()
        self.n_features = n_features
        self.n_stocks = n_stocks
        self.is_volumes = is_volumes
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

        self.linear_noise = nn.Linear(n_features + 1, n_features)

        self.rnn_encoder = RNNEncoder(
            n_features=n_features + 1,
            encoder_length=encoder_length,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
        )

        self.rnn_decoder = RNNDecoder(
            n_features=n_features,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
        )

        self.linear_out = nn.Linear(hidden_dim, n_features)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        o = torch.zeros_like(x)[:, :, : self.decoder_length]
        o = torch.permute(o, (0, 2, 1))

        # Noise injection in encoder's input
        encoder_input = torch.cat((x, noise), dim=1)
        encoder_input = torch.permute(encoder_input, (0, 2, 1))

        # Encoding
        _, encoder_hidden = self.rnn_encoder(encoder_input)

        # Decoding
        prev_hidden = encoder_hidden.unsqueeze(0)
        y_prev = x[:, :, -1].unsqueeze(1)
        for i in range(self.decoder_length):
            y_prev, prev_hidden = self.rnn_decoder(y_prev, prev_hidden)
            o[:, i] = y_prev.squeeze(1)
        o = torch.permute(o, (0, 2, 1))

        # Handling volumes
        if self.is_volumes:
            o_price, o_volume = o[:, : self.n_stocks, :], o[:, self.n_stocks :, :]
            o_volume = self.tanh(o_volume)
            o = torch.concatenate((o_price, o_volume), dim=1)

        return o
