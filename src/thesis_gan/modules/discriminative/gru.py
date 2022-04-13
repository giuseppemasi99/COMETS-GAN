import torch
import torch.nn as nn


class GRUDiscriminative(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int) -> None:
        super(GRUDiscriminative, self).__init__()
        self.gru = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.gru(x)
        o = self.tanh(o)
        # Take last state
        o = o[:, -1, :]
        return self.linear(o)
