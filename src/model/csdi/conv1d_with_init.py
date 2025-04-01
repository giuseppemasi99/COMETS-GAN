import torch
import torch.nn as nn


class Conv1dWithInit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Conv1d(in_channels, out_channels, 4, padding="same")
        nn.init.kaiming_normal_(self.layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
