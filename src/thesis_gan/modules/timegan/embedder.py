import torch.nn as nn


class Embedder(nn.Module):
    def __init__(
        self,
        n_features: int,
        dropout: float,
        hidden_dim=None,
    ) -> None:
        super(Embedder, self).__init__()
        # e_X: H_X x X -> H_X
        # h_t = e_X(h_{t-1}, x_t)

        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True, dropout=dropout, bidirectional=False
        )

    def forward(self, encoder_sequence):
        # encoder_sequence.shape = [batch_size, encoder_length, n_features]

        o, _ = self.gru(encoder_sequence)
        # o.shape = [batch_size, sequence_length, hidden_size]

        return o
