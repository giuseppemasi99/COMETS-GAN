import torch


def non_linearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
