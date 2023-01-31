from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from thesis_gan.data.dataset import StockDataset
from thesis_gan.data.pipeline import Pipeline


class StockDatasetTimeGAN(StockDataset):
    def __init__(
        self,
        path: Path,
        target_feature_price: str,
        target_feature_volume: str,
        stock_names: List[str],
        sequence_length: int,
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

        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return ((len(self.data) - self.sequence_length) // self.stride) + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_slice = slice(self.stride * index, self.stride * index + self.sequence_length)
        x = torch.as_tensor(self.data[x_slice].T, dtype=torch.float)

        return_dict = dict(x=x)

        if self.target_feature_price is not None:
            x_prices = torch.as_tensor(self.prices[x_slice].T, dtype=torch.float)
            return_dict["prices"] = x_prices

        if self.target_feature_volume is not None:
            x_volumes = torch.as_tensor(self.volumes[x_slice].T, dtype=torch.float)
            return_dict["volumes"] = x_volumes

        return return_dict


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """

    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

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
