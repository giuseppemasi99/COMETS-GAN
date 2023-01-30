import torch.nn as nn


class Recoverer(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim=None,
    ) -> None:
        super(Recoverer, self).__init__()
        # r_X: H_X -> X
        # x^hat_t = r_X(h_t)

        self.feedforward = nn.Linear(hidden_dim, n_features)

    def forward(self, h_t):
        # h_t.shape = [batch_size, hidden_dim]

        o = self.feedforward(h_t)
        # o.shape = [batch_size, n_features]

        return o
