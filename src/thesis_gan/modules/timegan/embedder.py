import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(
        self,
        n_features: int,
        dropout: float,
        hidden_size=None,
        num_layers=None,
    ) -> None:
        super(Embedder, self).__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = [batch_size, n_features, sequence_length]

        x = torch.permute(x, (0, 2, 1))
        # x.shape = [batch_size, sequence_length, n_features]

        h, _ = self.gru(x)
        # h.shape = [batch_size, sequence_length, hidden_size]

        h = self.fc(h)
        # h.shape = [batch_size, sequence_length, hidden_size]

        h = self.sigmoid(h)

        return h
