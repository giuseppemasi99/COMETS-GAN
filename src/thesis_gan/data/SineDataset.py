from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from nn_core.nn_types import Split


def sine_data_generation(n_samples: int, seq_len: int, n_features: int, seed: Optional[int] = None) -> np.ndarray:
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate sine data
    data = []
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
    def __init__(
        self,
        n_samples: int,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        split: Split,
        seed: Optional[int] = None,
    ) -> None:
        super(SineDataset, self).__init__()
        seq_len = encoder_length + decoder_length
        data = sine_data_generation(n_samples, seq_len, n_features, seed)
        self.data = torch.as_tensor(data).permute(0, 2, 1)
        self.n_samples = n_samples
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_features = n_features
        self.split = split

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = self.data[index][..., : self.encoder_length]
        y = self.data[index][..., self.encoder_length : self.encoder_length + self.decoder_length]
        return {"x": x, "y": y}

    def __repr__(self) -> str:
        props = f"{self.split=}, {self.n_samples=}, {self.encoder_length=}, {self.decoder_length=}, {self.n_features=}"
        props = props.replace("self.", "")
        return f"SineDataset({props})"
