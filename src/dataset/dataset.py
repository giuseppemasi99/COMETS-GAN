from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.pipeline import Pipeline


class StockDataset(Dataset):
    def __init__(
        self, path: str, stock_names: List[str],
        target_feature_price: str, target_feature_volume: str,
        pipeline_price: Pipeline, pipeline_volume: Pipeline,
        encoder_length: int = None, decoder_length: int = None, stride: int = None,
    ) -> None:
        super().__init__()

        self.target_feature_price = target_feature_price
        self.target_feature_volume = target_feature_volume
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.stride = stride

        self.df = pd.read_csv(path, index_col=0)

        data_price = None
        if target_feature_price is not None:
            targets_price = [f"{target_feature_price}_{stock}" for stock in stock_names]
            self.prices = self.df[targets_price].to_numpy()
            data_price = pipeline_price.preprocess(self.prices)

        data_volume = None
        if target_feature_volume is not None:
            targets_volume = [f"{target_feature_volume}_{stock}" for stock in stock_names]
            self.volumes = self.df[targets_volume].to_numpy()
            data_volume = pipeline_volume.preprocess(self.volumes)

        if target_feature_price is not None and target_feature_volume is not None:
            self.data = np.concatenate((data_price, data_volume), axis=1)
        elif target_feature_price is not None:
            self.data = data_price
        elif target_feature_volume is not None:
            self.data = data_volume

    def __len__(self) -> int:
        if self.encoder_length is None:
            return 1
        return ((len(self.data) - (self.encoder_length + self.decoder_length)) // self.stride) + 1
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.encoder_length is None:
            x = torch.as_tensor(self.data.T, dtype=torch.float)
            return_dict = dict(x=x)
        else:
            x_slice = slice(self.stride * index, self.stride * index + self.encoder_length)
            y_slice = slice(
                self.stride * index + self.encoder_length,
                self.stride * index + self.encoder_length + self.decoder_length,
            )
            x = torch.as_tensor(self.data[x_slice].T, dtype=torch.float)
            y = torch.as_tensor(self.data[y_slice].T, dtype=torch.float)
            return_dict = dict(x=x, y=y)
        return return_dict