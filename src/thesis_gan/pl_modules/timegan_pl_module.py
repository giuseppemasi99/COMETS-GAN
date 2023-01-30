import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

import thesis_gan  # noqa
from thesis_gan.data.datamodule import MetaData

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        if self.hparams.dataset_type == "multistock":
            self.hparams.n_stocks = len(self.hparams.stock_names)
            self.feature_names = list()
            if self.hparams.target_feature_price is not None:
                self.feature_names.extend(
                    [stock_name + "_" + self.hparams.target_feature_price for stock_name in self.hparams.stock_names]
                )
            if self.hparams.target_feature_volume is not None:
                self.feature_names.extend(
                    [stock_name + "_" + self.hparams.target_feature_volume for stock_name in self.hparams.stock_names]
                )
            self.hparams.n_features = len(self.feature_names)

        self.embedder = hydra.utils.instantiate(
            self.hparams.embedder,
            n_features=self.hparams.n_features,
            _recursive_=False,
        )

        self.recoverer = hydra.utils.instantiate(
            self.hparams.recoverer,
            n_features=self.hparams.n_features,
            _recursive_=False,
        )

        self.pipeline_price = metadata.data_pipeline_price
        self.pipeline_volume = metadata.data_pipeline_volume

    def forward(self):
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def validation_epoch_end(self, samples: Sequence[Dict]) -> None:
        self.__validation_n_test_epoch_end(samples)

    def test_epoch_end(self, samples: Sequence[Dict]) -> None:
        self.__validation_n_test_epoch_end(samples)

    def __validation_n_test_epoch_end(self, samples: Sequence[Dict]) -> None:

        # Aggregation of the batches
        sequence, prices, volumes = list(), list(), list()
        for batch in samples:
            sequence.append(batch["sequence"])
            if self.hparams.target_feature_price is not None:
                prices.append(batch["prices"])
            if self.hparams.target_feature_volume is not None:
                volumes.append(batch["volumes"])

    def configure_optimizers(self) -> Tuple[Dict[str, Optimizer], Dict[str, Optimizer]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt_embedder = hydra.utils.instantiate(
            self.hparams.optimizer_embedder,
            params=self.embedder.parameters(),
            _convert_="partial",
        )
        opt_recoverer = hydra.utils.instantiate(
            self.hparams.optimizer_recoverer,
            params=self.recoverer.parameters(),
            _convert_="partial",
        )

        return (
            {"optimizer": opt_embedder, "frequency": 1},
            {"optimizer": opt_recoverer, "frequency": 1},
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        # optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
