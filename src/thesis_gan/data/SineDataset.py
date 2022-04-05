import numpy as np
import torch
from torch.utils.data import Dataset

from nn_core.nn_types import Split


def sine_data_generation(n_samples: int, seq_len: int, n_features: int) -> np.ndarray:
    data = []

    # Generate sine data
    for i in range(n_samples):
        # Initialize each time-series
        temp = []
        # For each feature
        for k in range(n_features):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return np.array(data)


class SineDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, n_features: int, split: Split) -> None:
        super(SineDataset, self).__init__()
        data = sine_data_generation(n_samples, seq_len, n_features)
        self.data = torch.as_tensor(data).permute(0, 2, 1)
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_features = n_features
        self.split = split

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> None:
        return self.data[index]

    def __repr__(self) -> str:
        return f"SineDataset({self.split=}, {self.n_samples=}, {self.seq_len=}, {self.n_features=})"
