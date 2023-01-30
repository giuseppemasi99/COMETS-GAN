import logging
import math
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

import thesis_gan  # noqa
from thesis_gan.common.metrics import (
    get_correlation_distances_dict,
    get_correlations_dict,
    get_metrics_listdict,
    get_plot_sf_absence_autocorrelation,
    get_plot_sf_aggregational_gaussianity,
    get_plot_sf_returns_distribution,
    get_plot_sf_volatility_clustering,
    get_plot_sf_volume_volatility_correlation,
    get_plot_timeseries,
)
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

        self.pipeline_price = metadata.data_pipeline_price
        self.pipeline_volume = metadata.data_pipeline_volume

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
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
                self.log_correlation_distances(y_real, y_pred, stage="train")
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

        # Building the whole real sequences
        sequence = torch.concatenate(sequence, dim=2)
        dict_with_reals = dict(sequence=sequence.detach().cpu())
        sequence = sequence[:, :, self.hparams.step_to_skip :]
        if self.hparams.target_feature_price is not None:
            prices = torch.concatenate(prices, dim=2)
            prices = prices[:, :, self.hparams.step_to_skip :]
            dict_with_reals["prices"] = prices.detach().cpu()
        if self.hparams.target_feature_volume is not None:
            volumes = torch.concatenate(volumes, dim=2)
            volumes = volumes[:, :, self.hparams.step_to_skip :]
            dict_with_reals["volumes"] = volumes.detach().cpu()

        # Autoregressive prediction
        dict_with_preds: Dict[str, torch.Tensor] = self.predict_autoregressively(
            sequence, prices, volumes, prediction_length=sequence.shape[2] - self.hparams.encoder_length
        )

        # Saving preds and reals in pickle files
        if hasattr(self.logger, "run_dir"):
            if not os.path.exists(self.logger.run_dir):
                os.makedirs(self.logger.run_dir)

            path_file_preds = f"{self.logger.run_dir}/preds_epoch={self.current_epoch}-target_price={self.hparams.target_feature_price}-target_volume={self.hparams.target_feature_volume}.pickle"
            with open(path_file_preds, "wb") as handle:
                pickle.dump(dict_with_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

            path_file_reals = f"{self.logger.run_dir}/reals.pickle"
            if self.hparams.save_reals is True and not os.path.exists(path_file_reals):
                with open(path_file_reals, "wb") as handle:
                    pickle.dump(dict_with_reals, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Retrieving the pred_sequence and the real sequence
        pred_sequence = dict_with_preds["pred_sequence"][:, :, : sequence.shape[2]]
        pred_sequence = pred_sequence.detach().cpu()
        sequence = sequence.detach().cpu()

        # Logging the correlations metrics
        self.log_correlations(pred_sequence, "pred")
        self.log_correlations(sequence, "real")
        self.log_correlation_distances(sequence, pred_sequence, "val")

        # Squeezing the batch dimension that is 1 at prediction time
        pred_sequence = pred_sequence.squeeze()
        sequence = sequence.squeeze()

        if self.current_epoch > 0:

            pred_prices = None
            # If there are prices
            if self.hparams.target_feature_price is not None:
                pred_prices = dict_with_preds["pred_prices"][:, : sequence.shape[1]]
                pred_prices = pred_prices.detach().cpu().squeeze().numpy()
                prices = prices.detach().cpu().squeeze().numpy()

                # Plot prices
                self.log_plot_timeseries(prices, pred_prices, "Prices")

                # Plots stylised facts
                if self.hparams.dataset_type == "multistock":
                    self.log_plot_sf_returns_distribution(prices, pred_prices)
                    self.log_plot_sf_aggregational_gaussianity(prices, pred_prices)
                    self.log_plot_sf_absence_autocorrelation(prices, pred_prices)
                    self.log_plot_sf_volatility_clustering(prices, pred_prices)

            pred_volumes = None
            # If there are volumes
            if self.hparams.target_feature_volume is not None:
                pred_volumes = dict_with_preds["pred_volumes"][:, : sequence.shape[1]]
                pred_volumes = pred_volumes.squeeze().detach().cpu().numpy()
                volumes = volumes.squeeze().detach().cpu().numpy()

                # Plot volumes
                self.log_plot_timeseries(volumes, pred_volumes, "Volumes")

                # Logging volumes metrics
                self.log_metrics_volume(volumes, pred_volumes)

            # If there are both prices and volumes
            if self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
                sequence_price, pred_sequence_price = (
                    sequence[: self.hparams.n_stocks],
                    pred_sequence[: self.hparams.n_stocks],
                )
                sequence_volume, pred_sequence_volume = (
                    sequence[self.hparams.n_stocks :],
                    pred_sequence[self.hparams.n_stocks :],
                )

                # Plot stylised fact
                if self.hparams.dataset_type == "multistock":
                    self.log_plot_sf_volume_volatility_correlation(
                        sequence_price, pred_sequence_price, sequence_volume, pred_sequence_volume
                    )

    def predict_autoregressively(
        self,
        sequence: torch.Tensor,
        prices: torch.Tensor,
        volumes: torch.Tensor,
        prediction_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        if prediction_length is None:
            prediction_length = self.hparams.decoder_length

        prediction_iterations = math.ceil(prediction_length / self.hparams.decoder_length)

        pred_sequence = sequence[:, :, : self.hparams.encoder_length]
        for i in range(prediction_iterations):
            noise = torch.randn(1, 1, self.hparams.encoder_length, device=self.device)
            o = self(pred_sequence[:, :, -self.hparams.encoder_length :], noise)
            pred_sequence = torch.concatenate((pred_sequence, o), dim=2)

        pred_sequence = pred_sequence.detach().cpu().numpy()
        return_dict = dict(pred_sequence=torch.Tensor(pred_sequence))

        pred_sequence_price, last_price = None, None
        pred_sequence_volume, last_volume = None, None

        # If there are prices
        if self.hparams.target_feature_price is not None:
            last_price = prices[:, : self.hparams.n_stocks, self.hparams.encoder_length - 1].detach().cpu()
            pred_sequence_price = pred_sequence[0, : self.hparams.n_stocks, :]
            pred_prices = self.pipeline_price.inverse_transform(pred_sequence_price.T, last_price).T
            return_dict["pred_prices"] = torch.Tensor(pred_prices)

        # If there are both prices and volumes
        elif self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
            last_volume = volumes[:, :, self.hparams.encoder_length - 1].detach().cpu()
            pred_sequence_volume = pred_sequence[0, self.hparams.n_stocks :, :]

        # If there are only volumes
        elif self.hparams.target_feature_volume is not None:
            last_volume = volumes[:, : self.hparams.n_stocks, self.hparams.encoder_length - 1].detach().cpu()
            pred_sequence_volume = pred_sequence[0, : self.hparams.n_stocks, :]

        if self.hparams.target_feature_volume is not None:
            pred_volumes = self.pipeline_volume.inverse_transform(pred_sequence_volume.T, last_volume).T
            return_dict["pred_volumes"] = torch.Tensor(pred_volumes)

        return return_dict

    def log_correlations(self, y_realOpred: torch.Tensor, realOpred: str) -> None:
        d = get_correlations_dict(y_realOpred, realOpred, self.feature_names)
        self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_correlation_distances(self, y_real: torch.Tensor, y_pred: torch.Tensor, stage: str) -> None:
        d = get_correlation_distances_dict(y_real, y_pred, stage, self.feature_names)
        self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_plot_timeseries(
        self,
        real: np.ndarray,
        pred: np.ndarray,
        price_o_volume: str,
    ) -> None:
        fig = get_plot_timeseries(
            real, pred, self.hparams.dataset_type, self.hparams.stock_names, self.hparams.encoder_length, price_o_volume
        )
        title = f"{price_o_volume} - Epoch {self.current_epoch}"
        self.logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

    def log_plot_sf_returns_distribution(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
    ) -> None:
        fig = get_plot_sf_returns_distribution(
            prices,
            pred_prices,
            self.hparams.stock_names,
        )
        title = f"Returns distribution - Epoch {self.current_epoch}"
        self.logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

    def log_plot_sf_aggregational_gaussianity(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
    ) -> None:
        for stock_index, stock_name in enumerate(self.hparams.stock_names):
            fig = get_plot_sf_aggregational_gaussianity(
                prices[stock_index],
                pred_prices[stock_index],
                stock_name,
            )
            title = f"Distribution of returns with increased time scale - {stock_name} - Epoch {self.current_epoch}"
            self.logger.experiment.log({title: wandb.Image(fig)})
            plt.close(fig)

    def log_plot_sf_absence_autocorrelation(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
    ) -> None:
        for stock_index, stock_name in enumerate(self.hparams.stock_names):
            fig = get_plot_sf_absence_autocorrelation(
                prices[stock_index],
                pred_prices[stock_index],
                stock_name,
            )
            title = f"Returns Autocorrelations - {stock_name} - Epoch {self.current_epoch}"
            self.logger.experiment.log({title: wandb.Image(fig)})
            plt.close(fig)

    def log_plot_sf_volatility_clustering(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
    ) -> None:
        fig = get_plot_sf_volatility_clustering(
            prices,
            pred_prices,
            self.hparams.stock_names,
        )
        title = f"Volatility Clustering - Epoch {self.current_epoch}"
        self.logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

    def log_plot_sf_volume_volatility_correlation(
        self, sequence_price, pred_sequence_price, sequence_volume, pred_sequence_volume
    ) -> None:
        fig = get_plot_sf_volume_volatility_correlation(
            sequence_price,
            pred_sequence_price,
            sequence_volume,
            pred_sequence_volume,
            self.hparams.stock_names,
            self.hparams.delta,
        )
        title = f"Volume-Volatility Correlation - Epoch {self.current_epoch}"
        self.logger.experiment.log({title: wandb.Image(fig)})
        plt.close(fig)

    def log_metrics_volume(self, ts_real: np.ndarray, ts_pred: np.ndarray) -> None:
        for d in get_metrics_listdict(ts_real, ts_pred, self.hparams.stock_names):
            self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

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
            # Line2D([0], [0], color="C0", lw=2, label="Observed"),
            # Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
            # Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
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
