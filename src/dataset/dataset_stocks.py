from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.preprocessing import Pipeline


class DatasetStocks(Dataset):
    
    def __init__(self, file_path_data: str, feature_names: List[str], seq_len: int, pipeline: Pipeline) -> None:
        super().__init__()
        self.feature_names = feature_names

        self.seq_len = seq_len

        self.data_unprocessed: np.ndarray = pd.read_csv(file_path_data, index_col=0)[feature_names].values

        self.data: np.ndarray = pipeline.preprocess(self.data_unprocessed)
        # self.data.shape = [time_series_len, n_features]

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        x_0 = self.data[index: index + self.seq_len]
        return dict(x_0=torch.as_tensor(x_0, dtype=torch.float))

    def __len__(self) -> int:
        return len(self.data) - self.seq_len
