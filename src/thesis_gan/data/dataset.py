from pathlib import Path
from typing import Dict, List

import hydra
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
        targets: List[str],
        encoder_length: int,
        decoder_length: int,
        stride: int,
        data_pipeline: Pipeline,
        split: Split,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.stride = stride
        self.data_pipeline = data_pipeline
        self.split = split

        # Preprocess dataset targets
        self.data = data_pipeline.preprocess(self.df, targets)
        # Keep non preprocessed prices
        self.prices = self.df[targets].to_numpy()

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

        return {"x": x, "y": y, "x_prices": x_prices}

    def __repr__(self) -> str:
        return f"StockDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    pipeline = hydra.utils.instantiate(cfg.nn.data.data_pipeline)
    _: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train, split="train", data_pipeline=pipeline, _recursive_=False
    )


if __name__ == "__main__":
    main()
