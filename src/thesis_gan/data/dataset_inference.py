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


class StockDatasetInference(StockDataset):
    def __init__(
        self,
        path: Path,
        target_feature_price: str,
        target_feature_volume: str,
        stock_names: List[str],
        data_pipeline_price: Pipeline,
        data_pipeline_volume: Pipeline,
        split: Split,
    ) -> None:
        super(StockDatasetInference, self).__init__(
            path,
            target_feature_price,
            target_feature_volume,
            stock_names,
            data_pipeline_price,
            data_pipeline_volume,
            split,
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = torch.as_tensor(self.data.T, dtype=torch.float)
        return_dict = dict(x=x)

        if self.target_feature_price is not None:
            prices = torch.as_tensor(self.prices.T, dtype=torch.float)
            return_dict["prices"] = prices

        if self.target_feature_volume is not None:
            volumes = torch.as_tensor(self.volumes.T, dtype=torch.float)
            return_dict["volumes"] = volumes

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

    data_pipeline_price = hydra.utils.instantiate(cfg.data.module.data_pipeline_price)
    data_pipeline_volume = hydra.utils.instantiate(cfg.data.module.data_pipeline_volume)

    _: Dataset = hydra.utils.instantiate(
        cfg.data.module.datasets.val[0],
        split="val",
        data_pipeline_price=data_pipeline_price,
        data_pipeline_volume=data_pipeline_volume,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
