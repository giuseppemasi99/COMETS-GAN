from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dataset.pipeline import Pipeline


class StockDataset(Dataset):
    def __init__(
        self, path: str, stock_names: List[str],
        target_feature_price: str, target_feature_volume: str,
        pipeline_price: Pipeline, pipeline_volume: Pipeline,
        encoder_length: int = None, decoder_length: int = None, 
        stride: int = None, generation_length: int = None
    ) -> None:
        super().__init__()

        self.target_feature_price = target_feature_price
        self.target_feature_volume = target_feature_volume
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.stride = stride
        self.generation_length = generation_length

        df = pd.read_csv(path, index_col=0)

        targets_price = [f"{target_feature_price}_{stock}" for stock in stock_names]
        self.prices: np.ndarray = df[targets_price].to_numpy()
        data_price: np.ndarray = pipeline_price.preprocess(self.prices)

        targets_volume = [f"{target_feature_volume}_{stock}" for stock in stock_names]
        self.volumes: np.ndarray = df[targets_volume].to_numpy()
        data_volume: np.ndarray = pipeline_volume.preprocess(self.volumes)

        self.data: np.ndarray = np.concatenate((data_price, data_volume), axis=1)

        def to_bins(x):
            start_seconds = x.hour * 3600 + x.minute * 60 + x.second
            ## midnight 34200
            start_seconds -= 34200 ## start at 9:30  
            bin = start_seconds // 600
            return bin / 40.

        df["h_time"] = pd.to_datetime(df.index)
        df["dt_timestep"] = df["h_time"].apply(lambda x: to_bins(x))
        self.t = df["dt_timestep"][1:].to_numpy()
