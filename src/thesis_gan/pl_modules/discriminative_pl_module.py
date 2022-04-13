import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger

from thesis_gan.data.datamodule import MetaData
from thesis_gan.modules.discriminative.gru import GRUDiscriminative

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        # Loss criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        self.model = GRUDiscriminative(n_features=self.hparams.n_features, hidden_dim=self.hparams.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        logits = self(x)
        y_hat = (torch.sigmoid(logits) > 0.5).int()
        loss = self.criterion(logits, y)
        return {"logits": logits.detach(), "y_hat": y_hat, "loss": loss}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Mapping[str, Any]:
        x = batch["x"]
        y = batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(step_out["y_hat"], y.int())
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x = batch["x"]
        y = batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(step_out["y_hat"], y.int())
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x = batch["x"]
        y = batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(step_out["y_hat"], y.int())
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
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
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]
