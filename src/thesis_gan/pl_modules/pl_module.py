import logging
import math
from itertools import combinations
from typing import Any, Dict, Optional, Sequence, Tuple

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
from thesis_gan.common.utils import autocorrelation, compute_avg_log_returns, compute_avg_volumes, corr
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
            self.hparams.n_features = (
                2 * self.hparams.n_stocks if self.hparams.target_feature_volume is not None else self.hparams.n_stocks
            )
            self.feature_names = [
                stock_name + "_" + self.hparams.target_feature_price for stock_name in self.hparams.stock_names
            ]
            if self.hparams.target_feature_volume is not None:
                self.feature_names.extend(
                    [stock_name + "_" + self.hparams.target_feature_volume for stock_name in self.hparams.stock_names]
                )

        self.generator = hydra.utils.instantiate(
            self.hparams.generator,
            n_features=self.hparams.n_features,
            n_stocks=self.hparams.n_stocks,
            is_volumes=True if self.hparams.target_feature_volume is not None else False,
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
            # x.shape [batch_size, num_features, encoder_length]
            # noise.shape = [batch_size, 1, encoder_length]
            y_pred = self(x, noise)
            g_loss = -torch.mean(self.discriminator(x, y_pred))
            self.log_dict({"loss/gen": g_loss}, on_step=True, on_epoch=True, prog_bar=True)
            if self.hparams.dataset_type == "multistock":
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

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def validation_epoch_end(self, samples: Sequence[Dict]) -> None:
        sequence, prices, volumes = list(), list(), list()
        for batch in samples:
            sequence.append(batch["sequence"])
            prices.append(batch["prices"])
            if self.hparams.target_feature_volume is not None:
                volumes.append(batch["volumes"])

        sequence = torch.concatenate(sequence, dim=2)
        prices = torch.concatenate(prices, dim=2)
        if self.hparams.target_feature_volume is not None:
            volumes = torch.concatenate(volumes, dim=2)

        dict_with_preds = self.predict_autoregressively(
            sequence, prices, volumes, prediction_length=sequence.shape[2] - self.hparams.encoder_length
        )

        pred_sequence = dict_with_preds["pred_sequence"]
        pred_prices = dict_with_preds["pred_prices"]

        self.log_correlation(sequence, "real")
        self.log_correlation(pred_sequence, "pred")

        pred_sequence = pred_sequence.squeeze().detach().cpu()
        pred_prices = pred_prices.squeeze().detach().cpu()
        sequence = sequence.squeeze().detach().cpu()
        prices = prices.squeeze().detach().cpu()

        sequence_price, pred_sequence_price = sequence[: self.hparams.n_stocks], pred_sequence[: self.hparams.n_stocks]
        self.log_multistock_prices(sequence_price, pred_sequence_price, prices, pred_prices)

        if self.hparams.target_feature_volume is not None:
            pred_volumes = dict_with_preds["pred_volumes"]
            pred_volumes = pred_volumes.squeeze().detach().cpu().numpy()
            volumes = volumes.squeeze().detach().cpu().numpy()

            sequence_volume, pred_sequence_volume = (
                sequence[self.hparams.n_stocks :],
                pred_sequence[self.hparams.n_stocks :],
            )
            self.log_multistock_volumes(sequence_volume, pred_sequence_volume, volumes, pred_volumes)
            self.log_metrics_volume(volumes, pred_volumes)
            self.log_multistock_minmax_volumes(volumes, pred_volumes)
            self.log_multistock_volume_volatility_correlation(
                sequence_price, pred_sequence_price, sequence_volume, pred_sequence_volume
            )

    def predict_autoregressively(
        self,
        sequence: torch.Tensor,
        prices: torch.Tensor,
        volumes: torch.Tensor,
        prediction_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        last_price = prices[:, :, self.hparams.encoder_length - 1].detach().cpu()

        if prediction_length is None:
            prediction_length = self.hparams.decoder_length

        prediction_iterations = math.ceil(prediction_length / self.hparams.decoder_length)

        pred_sequence = sequence[:, :, : self.hparams.encoder_length]
        for i in range(prediction_iterations):
            noise = torch.randn(1, 1, self.hparams.encoder_length, device=self.device)
            o = self(pred_sequence[:, :, -self.hparams.encoder_length :], noise)
            pred_sequence = torch.concatenate((pred_sequence, o), dim=2)

        pred_sequence = pred_sequence.detach().cpu().numpy()

        pred_prices = self.pipeline_price.inverse_transform(
            pred_sequence[0, : self.hparams.n_stocks, :].T, last_price
        ).T

        return_dict = dict(pred_sequence=torch.Tensor(pred_sequence), pred_prices=torch.Tensor(pred_prices))

        if self.hparams.target_feature_volume is not None:
            last_volume = volumes[:, :, self.hparams.encoder_length - 1].detach().cpu()
            pred_volumes = self.pipeline_volume.inverse_transform(
                pred_sequence[0, self.hparams.n_stocks :, :].T, last_volume
            ).T
            return_dict["pred_volumes"] = torch.Tensor(pred_volumes)

        return return_dict

    def log_correlation(self, y_realOpred: torch.Tensor, realOpred: str) -> None:
        correlations = corr(y_realOpred).squeeze()
        print("correlations.shape:", correlations.shape)

        metric_names = [f"{realOpred}_correlation/{'-'.join(x)}" for x in combinations(self.feature_names, 2)]

        self.log_dict(
            {metric: correlation.item() for metric, correlation in zip(metric_names, correlations)},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def log_correlation_distance(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> None:
        corr_real = corr(y_real)
        corr_pred = corr(y_pred)

        metric_names = [f"corr_dist/{'-'.join(x)}" for x in combinations(self.feature_names, 2)]

        corr_distances = self.mse(corr_real, corr_pred).mean(dim=0)
        self.log_dict(
            {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def log_multistock_prices(
        self, sequence: torch.Tensor, pred_sequence: torch.Tensor, prices: torch.Tensor, pred_prices: torch.Tensor
    ) -> None:

        history_indexes = np.arange(self.hparams.encoder_length)
        continuation_indexes = np.arange(self.hparams.encoder_length, prices.shape[-1])

        history = sequence[:, : self.hparams.encoder_length].T
        reals = sequence[:, self.hparams.encoder_length :].T
        preds = pred_sequence[:, self.hparams.encoder_length :].T
        history_and_reals = np.concatenate((history, reals), axis=0)
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
                prices[target_idx, : self.hparams.encoder_length],
                color="C0",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                prices[target_idx, self.hparams.encoder_length :],
                color="C1",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                pred_prices[target_idx, self.hparams.encoder_length :],
                color="C2",
            )

            # Returns distributions
            title = f"{stock_name} - Returns"
            ax[1, target_idx].set_title(title)

            sns.histplot(
                reals[:, target_idx],
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

            autocorr_real = autocorrelation(history_and_reals[:, target_idx])
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

            abs_autocorr_real = autocorrelation(np.abs(history_and_reals[:, target_idx]))
            abs_autocorr_preds = autocorrelation(np.abs(history_and_preds[:, target_idx]))

            ax[3, target_idx].plot(np.zeros(len(abs_autocorr_real)), color="black")
            ax[3, target_idx].plot(abs_autocorr_real, color="C1")
            ax[3, target_idx].plot(abs_autocorr_preds, color="C2")

        fig.legend(handles=legend_elements, loc="upper right", ncol=1)
        fig.suptitle(f"Epoch {self.current_epoch}")
        fig.tight_layout()
        # title = f"Prices - Epoch {self.current_epoch}"
        title = "Prices"
        self.logger.experiment.log({title: wandb.Image(fig)})

        plt.close(fig)

    def log_multistock_volumes(
        self, sequence: torch.Tensor, pred_sequence: torch.Tensor, volumes: torch.Tensor, pred_volumes: torch.Tensor
    ) -> None:

        history_indexes = np.arange(self.hparams.encoder_length)
        continuation_indexes = np.arange(self.hparams.encoder_length, volumes.shape[-1])

        history = sequence[:, : self.hparams.encoder_length].T
        reals = sequence[:, self.hparams.encoder_length :].T
        preds = pred_sequence[:, self.hparams.encoder_length :].T
        history_and_reals = np.concatenate((history, reals), axis=0)
        history_and_preds = np.concatenate((history, preds), axis=0)

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
                volumes[target_idx, : self.hparams.encoder_length],
                color="C0",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                volumes[target_idx, self.hparams.encoder_length :],
                color="C1",
            )
            ax[0, target_idx].plot(
                continuation_indexes,
                pred_volumes[target_idx, self.hparams.encoder_length :],
                color="C2",
            )

            # Autocorrelation distributions
            title = f"{stock_name} - Autocorrelation"
            ax[1, target_idx].set_title(title)

            autocorr_real = autocorrelation(history_and_reals[:, target_idx])
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
        fig.suptitle(f"Epoch {self.current_epoch}")
        fig.tight_layout()
        # title = f"Volumes - Epoch {self.current_epoch}"
        title = "Volumes"
        self.logger.experiment.log({title: wandb.Image(fig)})

        plt.close(fig)

    def log_metrics_volume(self, ts_real: np.ndarray, ts_pred: np.ndarray) -> None:

        stat_names = ["Mean", "Std", "Kurtosis", "Skew"]
        stat_funcs = [np.mean, np.std, stats.kurtosis, stats.skew]

        for stat_name, stat_func in zip(stat_names, stat_funcs):
            d = dict()

            metrics_real = stat_func(ts_real, axis=1)
            metric_names = [f"Real Volume: {stat_name}/{stock_name}" for stock_name in self.hparams.stock_names]
            d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

            metrics_pred = stat_func(ts_pred, axis=1)
            metric_names = [f"Pred Volume: {stat_name}/{stock_name}" for stock_name in self.hparams.stock_names]
            d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

            self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_multistock_minmax_volumes(self, ts_real: np.ndarray, ts_pred: np.ndarray) -> None:
        d = dict()

        metrics_real = ts_real.min(axis=1)
        metric_names = [f"Real Volume: Min/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = ts_pred.min(axis=1)
        metric_names = [f"Pred Volume: Min/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        metrics_real = ts_real.max(axis=1)
        metric_names = [f"Real Volume: Max/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = ts_pred.max(axis=1)
        metric_names = [f"Pred Volume: Max/{stock_name}" for stock_name in self.hparams.stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_multistock_volume_volatility_correlation(
        self,
        sequence_price: torch.Tensor,
        pred_sequence_price: torch.Tensor,
        sequence_volume: torch.Tensor,
        pred_sequence_volume: torch.Tensor,
    ) -> None:
        sequence_price = sequence_price.numpy().T
        pred_sequence_price = pred_sequence_price.numpy().T
        sequence_volume = sequence_volume.numpy().T
        pred_sequence_volume = pred_sequence_volume.numpy().T

        real_avg_log_returns = compute_avg_log_returns(sequence_price, self.hparams.delta)
        real_avg_volumes = compute_avg_volumes(sequence_volume, self.hparams.delta)

        pred_avg_log_returns = compute_avg_log_returns(pred_sequence_price, self.hparams.delta)
        pred_avg_volumes = compute_avg_volumes(pred_sequence_volume, self.hparams.delta)

        fig, ax = plt.subplots(2, self.hparams.n_stocks, figsize=(7 * self.hparams.n_stocks, 10))

        for target_idx in range(self.hparams.n_stocks):
            stock_name = self.hparams.stock_names[target_idx]

            # Real volume-volatility correlation
            title = f"{stock_name} - Real"
            ax[0, target_idx].set_title(title)
            ax[0, target_idx].scatter(
                real_avg_log_returns[target_idx],
                real_avg_volumes[target_idx],
                color="C0",
            )
            ax[0, target_idx].set_xlabel("Avg log-returns")
            ax[0, target_idx].set_ylabel("Avg log-volumes")

            # Pred volume-volatility correlation
            title = f"{stock_name} - Pred"
            ax[1, target_idx].set_title(title)
            ax[1, target_idx].scatter(
                pred_avg_log_returns[target_idx],
                pred_avg_volumes[target_idx],
                color="C1",
            )
            ax[1, target_idx].set_xlabel("Avg log-returns")
            ax[1, target_idx].set_ylabel("Avg log-volumes")

        fig.suptitle(f"Epoch {self.current_epoch}")
        fig.tight_layout()
        # title = f"Volume-Volatility Corr - Epoch {self.current_epoch}"
        title = "Volume-Volatility Corr"
        self.logger.experiment.log({title: wandb.Image(fig)})

        plt.close(fig)

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
