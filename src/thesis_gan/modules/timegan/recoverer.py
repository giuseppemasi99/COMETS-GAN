import torch
import torch.nn as nn


class Recoverer(nn.Module):
    def __init__(self, n_features: int, dropout: int, hidden_size=None, num_layers=None) -> None:
        super(Recoverer, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, h):
        # h.shape = [batch_size, sequence_length, hidden_dim]

        o, _ = self.gru(h)
        # o.shape = [batch_size, sequence_length, hidden_dim]

        x_reconstructed = self.fc(o)
        # x_reconstructed.shape = [batch_size, sequence_length, n_features]

        x_reconstructed = torch.permute(x_reconstructed, (0, 2, 1))
        # x_reconstructed.shape = [batch_size, n_features, sequence_length]

        return x_reconstructed
