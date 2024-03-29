"""Docstring."""
import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

import thesis_gan  # noqa
from thesis_gan.data.datamodule import MetaData
from thesis_gan.pl_modules.pl_module import PLModule

pylogger = logging.getLogger(__name__)


class MyLightningModule(PLModule):
    """Docstring."""

    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        """Docstring."""
        super().__init__(metadata, *args, **kwargs)

        self.generator = hydra.utils.instantiate(
            self.hparams.generator,
            n_features=self.hparams.n_features,
            n_stocks=self.hparams.n_stocks,
            is_prices=True if self.hparams.target_feature_price is not None else False,
            is_volumes=True if self.hparams.target_feature_volume is not None else False,
            _recursive_=False,
        )

        self.discriminator = hydra.utils.instantiate(
            self.hparams.discriminator,
            n_features=self.hparams.n_features,
            _recursive_=False,
        )

        # self.discriminator_loss_criterion = nn.MarginRankingLoss()

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Docstring."""
        # x.shape = [batch_size, n_features, encoder_length]
        out = self.generator(x, noise)
        # out.shape = [batch_size, n_features, decoder_length]
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """Docstring."""
        # batch.keys() = ['x', 'y']

        x, y_real = batch["x"], batch["y"]
        # x.shape [batch_size, n_features, encoder_length]
        # y_real.shape [batch_size, n_features, decoder_length]

        # Sample noise
        noise = torch.randn(x.shape[0], 1, self.hparams.encoder_length, device=self.device)
        # noise.shape = [batch_size, 1, encoder_length]

        # Train generator
        if optimizer_idx == 0:
            y_pred = self(x, noise)
            g_loss = -torch.mean(self.discriminator(x, y_pred))
            self.log_dict({"loss/generator": g_loss}, on_step=True, on_epoch=True, prog_bar=True)
            if self.hparams.dataset_type == "multistock" and self.hparams.n_stocks > 1:
                self.log_correlation_distances(y_real, y_pred, stage="train")
            return g_loss

        # Train discriminator
        elif optimizer_idx == 1:
            y_pred = self(x, noise)
            real_validity = self.discriminator(x, y_real)
            fake_validity = self.discriminator(x, y_pred)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            # d_loss = self.discriminator_loss_criterion(real_validity, fake_validity, torch.ones_like(real_validity))
            self.log_dict({"loss/discriminator": d_loss}, on_step=True, on_epoch=True, prog_bar=True)
            return d_loss

    def validation_n_test_epoch_end(self, samples: Sequence[Dict[str, torch.Tensor]]) -> None:
        """Docstring."""
        # Aggregation of the batches
        dict_with_reals: Dict[str, torch.Tensor] = self.aggregate_from_batches(samples)

        x = dict_with_reals["x"]
        # x.shape = [n_features, sequence_length]
        sequence_length = x.shape[1]

        # Autoregressive prediction
        dict_with_preds: Dict[str, torch.Tensor] = self.predict_autoregressively(
            x, prediction_length=sequence_length - self.hparams.encoder_length
        )

        self.continue_validation_n_test_epoch_end(dict_with_reals, dict_with_preds)

    def predict_autoregressively(
        self,
        x: torch.Tensor,
        prediction_length: Optional[int] = None,
        sigma_scaler=None,
        linear=False,
    ) -> Dict[str, torch.Tensor]:
        """Docstring."""
        if prediction_length is None:
            prediction_length = self.hparams.decoder_length

        prediction_iterations = math.ceil(prediction_length / self.hparams.decoder_length)

        x_hat = x[:, : self.hparams.encoder_length].unsqueeze(0)
        # x_hat.shape = [1, n_features, encoder_length]

        i_start, i_end = 4, 30

        for i in range(prediction_iterations):
            noise = torch.randn(1, 1, self.hparams.encoder_length, device=self.device)
            o = self(x_hat[:, :, -self.hparams.encoder_length :], noise)

            if sigma_scaler is not None and i in range(i_start, i_end):
                std_KO, _ = torch.std(o, dim=2)[0]
                o[0, 0] = o[0, 0] + sigma_scaler * std_KO

            if linear and i >= i_start:
                o[0, 0] = torch.ones(150, device=self.device) / 10

            # elif sigma_scaler is not None and i in range(i_end + 2*n_iterations, i_end + 3*n_iterations):
            #     std_KO, _ = torch.std(o, dim=2)[0]
            #     o[0, 0] = o[0, 0] - sigma_scaler * std_KO

            x_hat = torch.cat((x_hat, o), dim=2)

        x_hat = x_hat.squeeze(dim=0).detach().cpu()
        # x_hat.shape = [n_features, sequence_length]

        x_hat = x_hat[:, : self.hparams.encoder_length + prediction_length]
        # x_hat.shape = [n_features, prediction_length]

        return self.unpack(x_hat)

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
        opt_g = hydra.utils.instantiate(
            self.hparams.optimizer_g,
            params=self.generator.parameters(),
            _convert_="partial",
        )
        opt_d = hydra.utils.instantiate(
            self.hparams.optimizer_d,
            params=self.discriminator.parameters(),
            _convert_="partial",
        )

        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.hparams.n_critic},
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module,
        # optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
