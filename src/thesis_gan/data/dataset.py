from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from thesis_gan.data.pipeline import Pipeline


class StockDataset(Dataset):
    def __init__(
        self,
        path: Path,
        target_feature_price: str,
        target_feature_volume: str,
        stock_names: List[str],
        encoder_length: int,
        decoder_length: int,
        stride: int,
        fool_scaler: bool,
        data_pipeline_price: Pipeline,
        data_pipeline_volume: Pipeline,
        split: Split,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.stride = stride
        self.fool_scaler = fool_scaler
        self.data_pipeline_price = data_pipeline_price
        self.data_pipeline_volume = data_pipeline_volume
        self.split = split

        targets_price = [f"{target_feature_price}_{stock}" for stock in stock_names]
        targets_volume = [f"{target_feature_volume}_{stock}" for stock in stock_names]

        # Pre-processing prices
        data_price = data_pipeline_price.preprocess(self.df, targets_price)

        # Pre-processing volumes
        if self.fool_scaler:
            self.df = self.df.append(2 * self.df.max(), ignore_index=True)
        data_volume = data_pipeline_volume.preprocess(self.df, targets_volume)
        if self.fool_scaler:
            data_volume = data_volume[:-1]
            self.df.drop(self.df.tail(1).index, inplace=True)

        self.data = np.concatenate((data_price, data_volume), axis=1)

        # Keep non preprocessed data
        self.prices = self.df[targets_price].to_numpy()
        self.volumes = self.df[targets_volume].to_numpy()

    def __len__(self) -> int:
        # Length of dataset is similar to output size of convolution
        return ((len(self.data) - (self.encoder_length + self.decoder_length)) // self.stride) + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_slice = slice(self.stride * index, self.stride * index + self.encoder_length)
        y_slice = slice(
            self.stride * index + self.encoder_length,
            self.stride * index + self.encoder_length + self.decoder_length,
        )

        x = torch.as_tensor(self.data[x_slice].T, dtype=torch.float)
        y = torch.as_tensor(self.data[y_slice].T, dtype=torch.float)
        x_prices = torch.as_tensor(self.prices[x_slice].T, dtype=torch.float)
        x_volumes = torch.as_tensor(self.volumes[x_slice].T, dtype=torch.float)

        return {"x": x, "y": y, "x_prices": x_prices, "x_volumes": x_volumes}

    def __repr__(self) -> str:
        return f"StockDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    pipeline_price = hydra.utils.instantiate(cfg.nn.data.data_pipeline_price)
    pipeline_volume = hydra.utils.instantiate(cfg.nn.data.data_pipeline_volume)
    _: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        split="train",
        data_pipeline_price=pipeline_price,
        data_pipeline_volume=pipeline_volume,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
