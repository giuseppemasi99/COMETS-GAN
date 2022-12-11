import logging
import math
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from torch import nn
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

import thesis_gan  # noqa
from thesis_gan.common.utils import autocorrelation, corr
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
            self.hparams.n_features = 2 * self.hparams.n_stocks

        self.generator = hydra.utils.instantiate(
            self.hparams.generator,
            n_features=self.hparams.n_features,
            _recursive_=False,
        )

        self.discriminator = hydra.utils.instantiate(
            self.hparams.discriminator,
            n_features=self.hparams.n_features,
            _recursive_=False,
        )

        self.pipeline_price = metadata.data_pipeline_price
        self.pipeline_volume = metadata.data_pipeline_volume

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # x.shape = [batch_size, n_features, encoder_length]
        out = self.generator(x, noise)
        # out.shape = [batch_size, n_features, decoder_length]
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        x = batch["x"]
        y_real = batch["y"]

        # Sample noise
        noise = torch.randn(x.shape[0], 1, self.hparams.encoder_length, device=self.device)

        # Train generator
        if optimizer_idx == 0:
            y_pred = self(x, noise)
            g_loss = -torch.mean(self.discriminator(x, y_pred))
            self.log_dict({"loss/gen": g_loss}, on_step=True, on_epoch=True, prog_bar=True)
            if self.hparams.dataset_type == "multistock":
                self.log_correlation(y_real, "real")
                self.log_correlation(y_pred, "pred")
                self.log_correlation_distance(y_real, y_pred)
            return g_loss

        # Train discriminator
        elif optimizer_idx == 1:
            y_pred = self(x, noise)
            real_validity = self.discriminator(x, y_real)
            fake_validity = self.discriminator(x, y_pred)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log_dict({"loss/disc": d_loss}, on_step=True, on_epoch=True, prog_bar=True)
            return d_loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if batch_idx % self.hparams.val_log_freq == 0:
            noise = torch.randn(batch["x"].shape[0], 1, self.hparams.encoder_length, device=self.device)
            fake = self(batch["x"], noise)
            if self.hparams.dataset_type == "multistock":
                x, y = batch["x"], batch["y"]
                x_prices, x_volumes = x[:, : self.hparams.n_stocks, :], x[:, self.hparams.n_stocks :, :]
                y_prices, y_volumes = y[:, : self.hparams.n_stocks, :], y[:, self.hparams.n_stocks :, :]
                prices, volumes = batch["x_prices"], batch["x_volumes"]
                fake_prices, fake_volumes = fake[:, : self.hparams.n_stocks, :], fake[:, self.hparams.n_stocks :, :]
                self.log_multistock_prices(x_prices, y_prices, prices, fake_prices, batch_idx)
                self.log_multistock_volumes(x_volumes, y_volumes, volumes, fake_volumes, batch_idx)
            elif self.hparams.dataset_type == "sines" or self.hparams.dataset_type == "gaussian":
                self.log_sines_gaussian(batch, fake, batch_idx)

    def log_correlation(self, y_realOpred: torch.Tensor, realOpred: str) -> None:
        correlation = corr(y_realOpred)

        feature_names = [
            "KO_price",
            "PEP_price",
            "NVDA_price",
            "KSU_price",
            "KO_volume",
            "PEP_volume",
            "NVDA_volume",
            "KSU_volume",
        ]
        metric_names = [f"{realOpred}_avg_correlation/{'-'.join(x)}" for x in combinations(feature_names, 2)]
        avg_correlations = torch.mean(correlation, dim=0)

        self.log_dict(
            {metric: avg_correlation.item() for metric, avg_correlation in zip(metric_names, avg_correlations)},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

    def log_correlation_distance(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> None:
        corr_real = corr(y_real)
        corr_pred = corr(y_pred)

        feature_names = [
            "KO_price",
            "PEP_price",
            "NVDA_price",
            "KSU_price",
            "KO_volume",
            "PEP_volume",
            "NVDA_volume",
            "KSU_volume",
        ]
        metric_names = [f"corr_dist/{'-'.join(x)}" for x in combinations(feature_names, 2)]

        corr_distances = self.mse(corr_real, corr_pred).mean(dim=0)
        self.log_dict(
            {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

    def log_multistock_prices(
        self, x: torch.Tensor, y: torch.Tensor, prices: torch.Tensor, y_pred: torch.Tensor, batch_idx: int
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

        real_prices = self.pipeline_price.inverse_transform(real, last_prices).T
        preds_prices = self.pipeline_price.inverse_transform(preds, last_prices).T

        history_and_real = np.concatenate((history, real), axis=0)
        history_and_preds = np.concatenate((history, preds), axis=0)

        fig, ax = plt.subplots(4, self.hparams.n_stocks, figsize=(7 * self.hparams.n_stocks, 10))
        legend_elements = [
            Line2D([0], [0], color="C0", lw=2, label="Observed"),
            Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
            Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
        ]

        for target_idx in range(self.hparams.n_stocks):
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
        title = f"Prices - Epoch {self.current_epoch} ({batch_idx})"
        self.logger.experiment.log({title: wandb.Image(fig)})

        plt.close(fig)

    def log_multistock_volumes(
        self, x: torch.Tensor, y: torch.Tensor, volumes: torch.Tensor, y_pred: torch.Tensor, batch_idx: int
    ) -> None:
        history_indexes = np.arange(self.hparams.encoder_length)
        continuation_indexes = np.arange(
            self.hparams.encoder_length,
            self.hparams.encoder_length + self.hparams.decoder_length,
        )

        history = x[0].detach().cpu().numpy().T
        real = y[0].detach().cpu().numpy().T
        preds = y_pred[0].detach().cpu().numpy().T

        history_volumes = volumes[0].detach().cpu().numpy()

        history_and_real = np.concatenate((history, real), axis=0)
        history_and_preds = np.concatenate((history, preds), axis=0)

        real_volumes = self.pipeline_volume.inverse_transform(real)
        preds_volumes = self.pipeline_volume.inverse_transform(preds)

        history_and_real_volumes = np.concatenate((history_volumes.T, real_volumes), axis=0)
        history_and_preds_volumes = np.concatenate((history_volumes.T, preds_volumes), axis=0)

        real_volumes = real_volumes.T
        preds_volumes = preds_volumes.T

        fig, ax = plt.subplots(2, self.hparams.n_stocks, figsize=(7 * self.hparams.n_stocks, 10))
        legend_elements = [
            Line2D([0], [0], color="C0", lw=2, label="Observed"),
            Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
            Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
        ]

        for target_idx in range(self.hparams.n_stocks):
            stock_name = self.hparams.stock_names[target_idx]

            # Plot of volumes
            title = f"{stock_name} - Volume"
            ax[0, target_idx].set_title(title)

            ax[0, target_idx].plot(
                history_indexes,
                history_volumes[target_idx],
                color="C0",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                real_volumes[target_idx],
                color="C1",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                preds_volumes[target_idx],
                color="C2",
            )

            # Autocorrelation distributions
            title = f"{stock_name} - Autocorrelation"
            ax[1, target_idx].set_title(title)

            autocorr_real = autocorrelation(history_and_real[:, target_idx])
            autocorr_preds = autocorrelation(history_and_preds[:, target_idx])
            sns.histplot(
                autocorr_real,
                color="C1",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[1, target_idx],
            )
            sns.histplot(
                autocorr_preds,
                color="C2",
                kde=True,
                stat="density",
                linewidth=0,
                ax=ax[1, target_idx],
            )

        fig.legend(handles=legend_elements, loc="upper right", ncol=1)
        fig.tight_layout()
        title = f"Volumes - Epoch {self.current_epoch} ({batch_idx})"
        self.logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

        self.log_multistock_minmax_volumes(history_and_real_volumes, history_and_preds_volumes)

        stat_names = ["Mean", "Std", "Kurtosis", "Skew"]
        stat_funcs = [np.mean, np.std, stats.kurtosis, stats.skew]
        for stat_name, stat_func in zip(stat_names, stat_funcs):
            self.log_metrics_volume(history_and_real_volumes, history_and_preds_volumes, stat_name, stat_func)

    def log_metrics_volume(
        self,
        ts_real: np.ndarray,
        ts_pred: np.ndarray,
        statistic_name: str,
        statistical_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        d = dict()

        metrics_real = statistical_func(ts_real, axis=0)
        metric_names = [f"Real Volume: {statistic_name}/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = statistical_func(ts_pred, axis=0)
        metric_names = [f"Pred Volume: {statistic_name}/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        self.log_dict(d, on_step=True, on_epoch=True, prog_bar=False)

    def log_multistock_minmax_volumes(self, ts_real: np.ndarray, ts_pred: np.ndarray) -> None:
        d = dict()

        metrics_real = ts_real.min(axis=0)
        metric_names = [f"Real Volume: Min/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = ts_pred.min(axis=0)
        metric_names = [f"Pred Volume: Min/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        metrics_real = ts_real.mac(axis=0)
        metric_names = [f"Real Volume: Max/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = ts_pred.max(axis=0)
        metric_names = [f"Pred Volume: Max/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        self.log_dict(d, on_step=True, on_epoch=True, prog_bar=False)

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

    def inverse_transform(self, y: torch.Tensor, last_prices: torch.Tensor) -> np.ndarray:
        print("INVERSE_TRANSFORM")
        y_prices = []
        for i in range(len(last_prices)):
            y_prices.append(self.pipeline.inverse_transform(y[i].detach().cpu().numpy().T, last_prices[i]).T)

        return np.stack(y_prices)

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        batch_size, _, encoder_length = batch["x"].shape
        noise = torch.randn(batch_size, 1, encoder_length)

        y_pred = self(batch["x"], noise)
        last_prices = batch["x_prices"][:, :, -1]
        y_pred_prices = self.inverse_transform(y_pred, last_prices)

        return_dict = dict(y_pred=y_pred, y_pred_prices=y_pred_prices)

        return return_dict

    def predict_autoregressively(
        self, batch: Dict[str, torch.Tensor], prediction_length: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        assert batch["x"].shape[0] == 1
        noise = torch.randn(1, 1, self.hparams.encoder_length)
        last_prices = batch["x_prices"][:, :, -1]

        if prediction_length is None:
            prediction_length = self.hparams.decoder_length

        prediction_iterations = math.ceil(prediction_length / self.hparams.decoder_length)

        y_pred = []
        batch_ = batch.copy()
        for i in range(prediction_iterations):
            o = self(batch_["x"], noise)
            y_pred.append(o)

            batch_["x"] = torch.cat((batch_["x"], o), dim=-1)[..., self.hparams.decoder_length :]

        y_pred = torch.cat(y_pred, dim=-1)[..., :prediction_length]
        y_pred_prices = self.pipeline.inverse_transform(y_pred[0].detach().cpu().numpy().T, last_prices).T

        return_dict = dict(y_pred=y_pred, y_pred_prices=y_pred_prices)

        return return_dict

    def log_sines_gaussian(self, batch: Dict[str, torch.Tensor], y_pred: torch.Tensor, batch_idx: int) -> None:
        history_indexes = np.arange(self.hparams.encoder_length)
        continuation_indexes = np.arange(
            self.hparams.encoder_length,
            self.hparams.encoder_length + self.hparams.decoder_length,
        )

        x = batch["x"]
        y = batch["y"]

        history = x[0].detach().cpu().numpy()
        real = y[0].detach().cpu().numpy()
        preds = y_pred[0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, self.hparams.n_features, figsize=(7 * self.hparams.n_features, 4))
        legend_elements = [
            Line2D([0], [0], color="C0", lw=2, label="Observed"),
            Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
            Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
        ]

        for target_idx in range(self.hparams.n_features):
            # Plot of prices
            title = f"Feature {target_idx}"
            ax[target_idx].set_title(title)

            ax[target_idx].plot(
                history_indexes,
                history[target_idx],
                color="C0",
            )
            ax[target_idx].plot(
                continuation_indexes,
                real[target_idx],
                color="C1",
            )
            ax[target_idx].plot(
                continuation_indexes,
                preds[target_idx],
                color="C2",
            )

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
