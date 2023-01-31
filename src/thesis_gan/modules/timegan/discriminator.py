import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        dropout: float,
        hidden_size=None,
        num_layers=None,
    ) -> None:
        super(Discriminator, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, h):
        # h.shape = [batch_size, sequence_length, hidden_size]

        o, _ = self.gru(h)
        # o.shape = [batch_size, sequence_length, 2*hidden_size]

        y = self.fc(o)
        # y.shape = [batch_size, sequence_length, 1]

        return y
