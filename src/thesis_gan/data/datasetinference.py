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


class StockDatasetInference(Dataset):
    def __init__(
        self,
        path: Path,
        target_feature_price: str,
        target_feature_volume: str,
        stock_names: List[str],
        encoder_length: int,
        decoder_length: int,
        data_pipeline_price: Pipeline,
        data_pipeline_volume: Pipeline,
        split: Split,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.target_feature_price = target_feature_price
        self.target_feature_volume = target_feature_volume
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.data_pipeline_price = data_pipeline_price
        self.data_pipeline_volume = data_pipeline_volume
        self.split = split

        if target_feature_price is not None:
            targets_price = [f"{target_feature_price}_{stock}" for stock in stock_names]
            data_price = data_pipeline_price.preprocess(self.df, targets_price)
            self.prices = self.df[targets_price].to_numpy()

        if target_feature_volume is not None:
            targets_volume = [f"{target_feature_volume}_{stock}" for stock in stock_names]
            data_volume = data_pipeline_volume.preprocess(self.df, targets_volume)
            self.volumes = self.df[targets_volume].to_numpy()

        if target_feature_price is not None and target_feature_volume is not None:
            self.data = np.concatenate((data_price, data_volume), axis=1)
        elif target_feature_price is not None:
            self.data = data_price
        elif target_feature_volume is not None:
            self.data = data_volume

    def __len__(self) -> int:
        return ((len(self.data) - self.encoder_length) // self.decoder_length) + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        if index == 0:
            sequence_slice = slice(index, self.encoder_length)
        else:
            sequence_slice = slice(
                self.encoder_length + (index - 1) * self.decoder_length,
                self.encoder_length + index * self.decoder_length,
            )

        data = torch.as_tensor(self.data[sequence_slice].T, dtype=torch.float)

        return_dict = dict(sequence=data)

        if self.target_feature_price is not None:
            prices = torch.as_tensor(self.prices[sequence_slice].T, dtype=torch.float)
            return_dict["prices"] = prices

        if self.target_feature_volume is not None:
            volumes = torch.as_tensor(self.volumes[sequence_slice].T, dtype=torch.float)
            return_dict["volumes"] = volumes

        return return_dict

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
        cfg.nn.data.datasets.val[0],
        split="val",
        data_pipeline_price=pipeline_price,
        data_pipeline_volume=pipeline_volume,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
