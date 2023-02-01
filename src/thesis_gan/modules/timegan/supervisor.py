import torch.nn as nn


class Supervisor(nn.Module):
    def __init__(self, dropout, hidden_size=None, num_layers=None) -> None:
        super(Supervisor, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers - 1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, h):
        # h.shape = [batch_size, sequence_length, hidden_size]

        o, _ = self.gru(h)
        # o.shape = [batch_size, sequence_length, hidden_size]

        s = self.fc(o)
        # s.shape = [batch_size, sequence_length, hidden_size]

        return s
