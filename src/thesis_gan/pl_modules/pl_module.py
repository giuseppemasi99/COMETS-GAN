import logging
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
from src.thesis_gan.data.datamodule import MetaData
from src.thesis_gan.modules.module import ConditionalDiscriminator, ConditionalGenerator

pylogger = logging.getLogger(__name__)


# TODO: remove
def autocorrelation(series: np.ndarray) -> np.ndarray:
    n = len(series)

    def r(h: float) -> float:
        return ((series[: n - h] - mean) * (series[h:] - mean)).sum() / n / c0

    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n
    x = np.arange(n) + 1
    y = np.array([r(loc) for loc in x])
    return y


# TODO: remove


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.generator = ConditionalGenerator(
            self.hparams.encoder_length,
            self.hparams.decoder_length,
            self.hparams.n_features,
            self.hparams.dropout,
            self.hparams.gen_hidden_dim,
        )
        self.discriminator = ConditionalDiscriminator(
            self.hparams.encoder_length,
            self.hparams.decoder_length,
            self.hparams.n_features,
            self.hparams.dropout,
            self.hparams.disc_hidden_dim,
        )

        self.pipeline = metadata.data_pipeline

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.generator(x, noise)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        x = batch["x"]
        y_real = batch["y"]

        # Sample noise
        noise = torch.randn(x.shape[0], 1, self.hparams.encoder_length, device=self.device)

        # Train generator
        if optimizer_idx == 0:
            y_pred = self(batch, noise)
            g_loss = -torch.mean(self.discriminator(batch, y_pred))
            self.log_dict({"loss/gen": g_loss}, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss

        # Train discriminator
        elif optimizer_idx == 1:
            y_pred = self(batch, noise)

            real_validity = self.discriminator(batch, y_real)
            fake_validity = self.discriminator(batch, y_pred)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log_dict({"loss/disc": d_loss}, on_step=True, on_epoch=True, prog_bar=True)
            return d_loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if batch_idx % self.hparams.val_log_freq:
            x = batch["x"]
            y = batch["y"]
            x_prices = batch["x_prices"]

            noise = torch.randn(x.shape[0], 1, self.hparams.encoder_length, device=self.device)
            fake = self(batch, noise)
            self.log_predictions(x, y, fake, x_prices, batch_idx)

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

        # opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        # if "lr_scheduler" not in self.hparams:
        #     return [opt]
        # scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        # return [opt], [scheduler]

    def log_predictions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        prices: torch.Tensor,
        batch_idx: int,
    ) -> None:
        history_indexes = np.arange(self.hparams.encoder_length)
        continuation_indexes = np.arange(
            self.hparams.encoder_length,
            self.hparams.encoder_length + self.hparams.decoder_length,
        )

        history = x[0].detach().cpu().numpy().T
        real = y[0].detach().cpu().numpy().T
        preds = y_pred[0].detach().cpu().numpy().T

        history_prices = prices[0].detach().cpu().numpy()
        last_prices = history_prices[:, -1]

        real_prices = self.pipeline.inverse_transform(real, last_prices).T
        preds_prices = self.pipeline.inverse_transform(preds, last_prices).T

        history_and_real = np.concatenate((history, real), axis=0)
        history_and_preds = np.concatenate((history, preds), axis=0)

        fig, ax = plt.subplots(4, self.hparams.n_features, figsize=(7 * self.hparams.n_features, 10))
        legend_elements = [
            Line2D([0], [0], color="C0", lw=2, label="Observed"),
            Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
            Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
        ]

        for target_idx in range(self.hparams.n_features):
            stock_name = self.hparams.stock_names[target_idx]

            # Plot of prices
            title = f"{stock_name} - Price"
            ax[0, target_idx].set_title(title)

            ax[0, target_idx].plot(
                history_indexes,
                history_prices[target_idx],
                color="C0",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                real_prices[target_idx],
                color="C1",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                preds_prices[target_idx],
                color="C2",
            )

            # Returns distributions
            title = f"{stock_name} - Returns"
            ax[1, target_idx].set_title(title)

            sns.histplot(
                real[:, target_idx],
                color="C1",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[1, target_idx],
            )
            sns.histplot(
                preds[:, target_idx],
                color="C2",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[1, target_idx],
            )

            # Autocorrelation distributions
            title = f"{stock_name} - Autocorrelation"
            ax[2, target_idx].set_title(title)

            autocorr_real = autocorrelation(history_and_real[:, target_idx])
            autocorr_preds = autocorrelation(history_and_preds[:, target_idx])
            sns.histplot(
                autocorr_real,
                color="C1",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[2, target_idx],
            )
            sns.histplot(
                autocorr_preds,
                color="C2",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[2, target_idx],
            )

            # Volatility clustering
            title = f"{stock_name} - Volatility clustering"
            ax[3, target_idx].set_title(title)

            abs_autocorr_real = autocorrelation(np.abs(history_and_real[:, target_idx]))
            abs_autocorr_preds = autocorrelation(np.abs(history_and_preds[:, target_idx]))

            ax[3, target_idx].plot(np.zeros(len(abs_autocorr_real)), color="black")
            ax[3, target_idx].plot(abs_autocorr_real, color="C1")
            ax[3, target_idx].plot(abs_autocorr_preds, color="C2")

        fig.legend(handles=legend_elements, loc="upper right", ncol=1)
        fig.tight_layout()
        title = f"Epoch {self.current_epoch} ({batch_idx})"
        self.logger.experiment.log({title: wandb.Image(fig)})

        plt.close(fig)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
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
