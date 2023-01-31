import logging
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


class PLModule(pl.LightningModule):
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

        self.pipeline_price = metadata.data_pipeline_price
        self.pipeline_volume = metadata.data_pipeline_volume

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        return batch

    def validation_epoch_end(self, samples: Sequence[Dict]) -> None:
        self.validation_n_test_epoch_end(samples)

    def test_epoch_end(self, samples: Sequence[Dict]) -> None:
        self.validation_n_test_epoch_end(samples)

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

    def save_files(self, dict_with_reals, dict_with_preds):
        if hasattr(self.logger, "run_dir"):
            if not os.path.exists(self.logger.run_dir):
                os.makedirs(self.logger.run_dir)

            path_file_preds = f"{self.logger.run_dir}/preds_timegan_epoch={self.current_epoch}-target_price={self.hparams.target_feature_price}-target_volume={self.hparams.target_feature_volume}.pickle"
            with open(path_file_preds, "wb") as handle:
                pickle.dump(dict_with_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

            path_file_reals = f"{self.logger.run_dir}/reals_timegan.pickle"
            if self.hparams.save_reals is True and not os.path.exists(path_file_reals):
                with open(path_file_reals, "wb") as handle:
                    pickle.dump(dict_with_reals, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
