from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import torch
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from thesis_gan.data.dataset import StockDataset
from thesis_gan.data.pipeline import Pipeline


class StockDatasetInference(StockDataset):
    def __init__(
        self,
        path: Path,
        target_feature_price: str,
        target_feature_volume: str,
        stock_names: List[str],
        encoder_length: int,
        decoder_length: int,
        stride: int,
        data_pipeline_price: Pipeline,
        data_pipeline_volume: Pipeline,
        split: Split,
    ) -> None:
        super().__init__(
            path,
            target_feature_price,
            target_feature_volume,
            stock_names,
            stride,
            data_pipeline_price,
            data_pipeline_volume,
            split,
        )

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

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

        return self.get_dict(sequence_slice)


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
