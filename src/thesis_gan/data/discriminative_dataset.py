from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from nn_core.nn_types import Split


class DiscriminativeDataset(Dataset):
    def __init__(
        self,
        path: Path,
        split: Split,
    ) -> None:
        super(DiscriminativeDataset, self).__init__()
        data = torch.load(path)
        self.data = data[:, :, :-1]
        self.labels = data[:, 0, -1:]
        self.split = split

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = self.data[index]
        y = self.labels[index]
        return {"x": x, "y": y}

    def __repr__(self) -> str:
        return f"DiscriminativeDataset({self.data.shape[0]})"
