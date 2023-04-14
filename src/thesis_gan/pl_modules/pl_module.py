import logging
import os
import pickle
from typing import Any, Dict, Optional, Sequence

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt

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
    get_plot_timeseries_conv,
    get_plot_timeseries_timegan,
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

        # TODO: controllare pipeline quando benchmark datasets
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

    def log_correlations(self, y: torch.Tensor, real_o_pred: str) -> None:
        d = get_correlations_dict(y, real_o_pred, self.hparams.feature_names)
        self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_correlation_distances(self, y_real: torch.Tensor, y_pred: torch.Tensor, stage: str) -> None:
        d = get_correlation_distances_dict(y_real, y_pred, stage, self.hparams.feature_names)
        self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def log_plot_timeseries(
        self,
        real: np.ndarray,
        pred: np.ndarray,
        price_o_volume: str,
    ) -> None:
        if self.hparams.model_type == "conv":
            fig = get_plot_timeseries_conv(
                real,
                pred,
                self.hparams.dataset_type,
                self.hparams.stock_names,
                self.hparams.encoder_length,
                price_o_volume,
            )
        else:

            if self.current_epoch < self.hparams.n_epochs_training_only_autoencoder:
                stage = "Only reconstruction"
            else:
                stage = "Joint"

            fig = get_plot_timeseries_timegan(
                real, pred, self.hparams.dataset_type, self.hparams.stock_names, price_o_volume, stage
            )
        title = f"{price_o_volume} - Epoch {self.current_epoch}"
        self.logger.experiment.log({title: wandb.Image(fig)})
        path = self.logger.experiment.dir + "/media/images/timeseries"
        self.__create_dirs_if_not_exist(path)
        plt.savefig(path + "/" + title + ".pdf")
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
        # self.logger.experiment.log({title: wandb.Image(fig)})
        path = self.logger.experiment.dir + "/media/images/returns_distribution"
        self.__create_dirs_if_not_exist(path)
        plt.savefig(path + "/" + title + ".pdf")
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
            # self.logger.experiment.log({title: wandb.Image(fig)})
            path = self.logger.experiment.dir + "/media/images/aggregational_gaussianity"
            self.__create_dirs_if_not_exist(path)
            plt.savefig(path + "/" + title + ".pdf")
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
            # self.logger.experiment.log({title: wandb.Image(fig)})
            path = self.logger.experiment.dir + "/media/images/absence_autocorrelation"
            self.__create_dirs_if_not_exist(path)
            plt.savefig(path + "/" + title + ".pdf")
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
        # self.logger.experiment.log({title: wandb.Image(fig)})
        path = self.logger.experiment.dir + "/media/images/volatility_clustering"
        self.__create_dirs_if_not_exist(path)
        plt.savefig(path + "/" + title + ".pdf")
        plt.close(fig)

    def log_plot_sf_volume_volatility_correlation(self, x_price, x_hat_price, x_volume, x_hat_volume) -> None:
        fig = get_plot_sf_volume_volatility_correlation(
            x_price,
            x_hat_price,
            x_volume,
            x_hat_volume,
            self.hparams.stock_names,
        )
        title = f"Volume-Volatility Correlation - Epoch {self.current_epoch}"
        # self.logger.experiment.log({title: wandb.Image(fig)})
        path = self.logger.experiment.dir + "/media/images/volume_volatility_correlation"
        self.__create_dirs_if_not_exist(path)
        plt.savefig(path + "/" + title + ".pdf")
        plt.close(fig)

    def log_metrics_volume(self, ts_real: np.ndarray, ts_pred: np.ndarray) -> None:
        for d in get_metrics_listdict(ts_real, ts_pred, self.hparams.stock_names):
            self.log_dict(d, on_step=False, on_epoch=True, prog_bar=False)

    def dict_with_tensors_to_numpy(self, d):
        for k, v in d.items():
            d[k] = v.numpy()
        return d

    def dict_with_tensors_cuda_to_cpu(self, d):
        for k, v in d.items():
            d[k] = v.cpu()
        return d

    def save_dict_in_pickle_file(self, d, file_name):
        if hasattr(self.logger, "run_dir"):
            s = self.logger.run_dir.split("/")
            run_id = s[-1]
            s = s[:-1]
            directory_diversity = "/".join(s) + "/diversity_val/" + run_id
            self.__create_dirs_if_not_exist(directory_diversity)

            path_file = f"{directory_diversity}/{file_name}"
            if not os.path.exists(path_file):
                with open(path_file, "wb") as handle:
                    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def aggregate_from_batches(self, samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        dict_with_reals = samples[0]

        dict_with_reals["x"] = dict_with_reals["x"].squeeze().detach()

        if self.hparams.target_feature_price is not None:
            dict_with_reals["prices"] = dict_with_reals["prices"].squeeze().detach()
        if self.hparams.target_feature_volume is not None:
            dict_with_reals["volumes"] = dict_with_reals["volumes"].squeeze().detach()

        return dict_with_reals

    def unpack(
        self,
        x_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        return_dict = dict(x_hat=x_hat)

        pred_x_volume = None

        # If there are prices
        if self.hparams.target_feature_price is not None:
            pred_x_price = x_hat[: self.hparams.n_stocks, :]
            pred_prices = self.pipeline_price.inverse_transform(pred_x_price.numpy().T).T
            return_dict["pred_prices"] = torch.Tensor(pred_prices)

        # If there are both prices and volumes
        if self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
            pred_x_volume = x_hat[self.hparams.n_stocks :, :]

        # If there are only volumes
        elif self.hparams.target_feature_volume is not None:
            pred_x_volume = x_hat

        # If there are volumes
        if self.hparams.target_feature_volume is not None:
            pred_volumes = self.pipeline_volume.inverse_transform(pred_x_volume.numpy().T).T
            return_dict["pred_volumes"] = torch.Tensor(pred_volumes)

        return return_dict

    def continue_validation_n_test_epoch_end(self, dict_with_reals, dict_with_preds) -> None:
        dict_with_reals = self.dict_with_tensors_cuda_to_cpu(dict_with_reals)

        x, x_hat = dict_with_reals["x"], dict_with_preds["x_hat"]
        # x.shape = [n_features, sequence_length]
        # x_hat.shape = [n_features, sequence_length]

        # Logging the correlations metrics
        self.log_correlations(x_hat, "pred")
        self.log_correlations(x, "real")
        self.log_correlation_distances(x, x_hat, "val")

        dict_with_reals: Dict[str, np.array] = self.dict_with_tensors_to_numpy(dict_with_reals)
        dict_with_preds: Dict[str, np.array] = self.dict_with_tensors_to_numpy(dict_with_preds)

        # Saving preds and reals in pickle files
        self.save_dict_in_pickle_file(
            dict_with_preds,
            file_name=f"preds_epoch={self.current_epoch}-"
            f"seed={self.hparams.seed}-"
            f"target_price={self.hparams.target_feature_price}-"
            f"target_volume={self.hparams.target_feature_volume}-"
            f"sampling_seed={np.random.get_state()[1][0]}"
            f".pickle",
        )

        if self.hparams.save_reals:
            self.save_dict_in_pickle_file(dict_with_reals, file_name="reals.pickle")

        if self.current_epoch > 0:
            self.do_plots(dict_with_reals, dict_with_preds)

    def do_plots(self, dict_with_reals: Dict[str, np.array], dict_with_preds: Dict[str, np.array]) -> None:

        x = dict_with_reals["x"]
        x_hat = dict_with_preds["x_hat"]

        # If there are prices
        if self.hparams.target_feature_price is not None:
            prices = dict_with_reals["prices"]
            pred_prices = dict_with_preds["pred_prices"]

            # Plot prices
            self.log_plot_timeseries(prices, pred_prices, "Prices")

            # Plots stylised facts
            if self.hparams.dataset_type == "multistock" and self.hparams.do_plot_stylised_facts:
                self.log_plot_sf_returns_distribution(prices, pred_prices)
                self.log_plot_sf_aggregational_gaussianity(prices, pred_prices)
                self.log_plot_sf_absence_autocorrelation(prices, pred_prices)
                self.log_plot_sf_volatility_clustering(prices, pred_prices)

        pred_volumes = None
        # If there are volumes
        if self.hparams.target_feature_volume is not None:
            volumes = dict_with_reals["volumes"]
            pred_volumes = dict_with_preds["pred_volumes"]

            # Plot volumes
            self.log_plot_timeseries(volumes, pred_volumes, "Volumes")

            # Logging volumes metrics
            self.log_metrics_volume(volumes, pred_volumes)

        # If there are both prices and volumes
        if self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
            x_price, x_hat_price = (
                x[: self.hparams.n_stocks],
                x_hat[: self.hparams.n_stocks],
            )
            x_volume, x_hat_volume = (
                x[self.hparams.n_stocks :],
                x_hat[self.hparams.n_stocks :],
            )

            # Plot stylised fact
            if self.hparams.dataset_type == "multistock" and self.hparams.do_plot_stylised_facts:
                self.log_plot_sf_volume_volatility_correlation(x_price, x_hat_price, x_volume, x_hat_volume)

    @staticmethod
    def __create_dirs_if_not_exist(path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)


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
