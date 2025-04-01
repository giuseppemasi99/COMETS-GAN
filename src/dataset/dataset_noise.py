import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetNoise(Dataset):

    def __init__(self, n_samples: int, seq_len: int, n_features: int) -> None:
        super().__init__()

        self.data: np.ndarray = np.random.randn(n_samples, seq_len, n_features)
        # self.data.shape = [n_samples, seq_len, n_features]

    def __getitem__(self, index) -> torch.Tensor:
        return torch.as_tensor(self.data[index], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.data)
