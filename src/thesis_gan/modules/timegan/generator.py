import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        dropout: float,
        hidden_size=None,
        num_layers=None,
    ) -> None:
        super(Generator, self).__init__()

        self.gru = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, noise):
        # noise.shape = [batch_size, noise_dim, sequence_length]

        noise = torch.permute(noise, (0, 2, 1))
        # noise.shape = [batch_size, sequence_length, noise_dim]

        o, _ = self.gru(noise)
        # o.shape = [batch_size, sequence_length, hidden_size]

        e_hat = self.fc(o)
        # o.shape = [batch_size, sequence_length, hidden_size]

        return e_hat
