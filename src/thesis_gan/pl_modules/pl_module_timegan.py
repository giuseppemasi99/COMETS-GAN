import logging
import math
from typing import Dict, Optional, Sequence

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

import thesis_gan  # noqa
from thesis_gan.data.datamodule import MetaData
from thesis_gan.pl_modules.pl_module import PLModule

pylogger = logging.getLogger(__name__)


class MyLightningModule(PLModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

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

        self.supervisor = hydra.utils.instantiate(
            self.hparams.supervisor,
            _recursive_=False,
        )

        self.generator = hydra.utils.instantiate(
            self.hparams.generator,
            n_features=self.hparams.noise_features,
            sequence_length=self.hparams.noise_features,
            _recursive_=False,
        )

        self.discriminator = hydra.utils.instantiate(
            self.hparams.discriminator,
            _recursive_=False,
        )

        self.mse = nn.MSELoss(reduction="mean")
        self.bcewl = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False

    def forward_embedder(self, x):
        # x.shape = [batch_size, n_features, sequence_length]
        h = self.embedder(x)
        # h.shape = [batch_size, sequence_length, hidden_size]
        return h

    def forward_recoverer(self, h):
        # h.shape = [batch_size, sequence_length, hidden_size]
        x_reconstructed = self.recoverer(h)
        # x_reconstructed.shape = [batch_size, n_features, sequence_length]
        return x_reconstructed

    def forward_supervisor(self, h):
        # h.shape = [batch_size, sequence_length, hidden_size]
        s = self.supervisor(h)
        # s.shape = [batch_size, sequence_length, hidden_size]
        return s

    def forward_autoencoder(self, x):
        # x.shape = [batch_size, n_features, sequence_length]
        x_tilde = self.forward_recoverer(self.forward_embedder(x))
        # x_tilde.shape = [batch_size, n_features, sequence_length]
        return x_tilde

    def forward_generator(self, batch_size):
        noise = torch.randn(batch_size, self.hparams.noise_dim, self.hparams.sequence_length, device=self.device)
        # noise.shape = [batch_size, noise_dim, sequence_length]
        e_hat = self.generator(noise)
        # e_hat.shape = [batch_size, sequence_length, hidden_size]
        return e_hat

    def forward_discriminator(self, h):
        # h_hat.shape = [batch_size, sequence_length, hidden_size]
        y = self.discriminator(h)
        # y.shape = [batch_size, sequence_length, 1]
        return y

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # batch.keys() = ['x', 'x_prices']

        opt_embedder, opt_recoverer, opt_supervisor, opt_generator, opt_discriminator = self.optimizers()
        opt_embedder.zero_grad()
        opt_recoverer.zero_grad()
        opt_supervisor.zero_grad()
        opt_generator.zero_grad()
        opt_discriminator.zero_grad()

        x = batch["x"]
        # x.shape = [batch_size, n_features, sequence_length]

        batch_size = x.shape[0]

        # Training with reconstruction loss only
        if self.current_epoch < self.hparams.n_epochs_training_only_reconstruction:
            x_tilde = self.forward_autoencoder(x)
            # x_tilde.shape = [batch_size, n_features, sequence_length]

            loss_reconstruction = 10 * torch.sqrt(self.__compute_loss_recontruction(x, x_tilde))

            self.log_dict({"loss/autoencoder": loss_reconstruction}, on_step=True, on_epoch=True, prog_bar=True)
            self.manual_backward(loss_reconstruction)

            opt_embedder.step()
            opt_recoverer.step()

        # Training with supervised loss only
        elif self.current_epoch < self.hparams.n_epochs_training_only_supervised:
            h = self.forward_embedder(x)
            # h.shape = [batch_size, sequence_length, hidden_size]
            h_hat_s = self.forward_supervisor(h)
            # h_hat_s.shape = [batch_size, sequence_length, hidden_size]

            loss_supervised = self.__compute_loss_supervised(h[:, 1:, :], h_hat_s[:, :-1, :])

            self.log_dict({"loss/supervisor": loss_supervised}, on_step=True, on_epoch=True, prog_bar=True)
            self.manual_backward(loss_supervised)

            opt_supervisor.step()

        # Joint training
        else:
            for _ in range(2):
                h = self.forward_embedder(x)
                # h.shape = [batch_size, sequence_length, hidden_size]

                h_hat_s = self.forward_supervisor(h)
                # h_hat_s.shape = [batch_size, sequence_length, hidden_size]

                e_hat = self.forward_generator(batch_size)
                # e_hat.shape = [batch_size, sequence_length, hidden_size]

                h_hat = self.forward_supervisor(e_hat)
                # h_hat.shape = [batch_size, sequence_length, hidden_size]

                x_hat = self.forward_recoverer(h_hat)
                # x_hat.shape = [batch_size, n_features, sequence_length]

                y_fake = self.forward_discriminator(h_hat)
                # y_fake.shape = [batch_size, sequence_length, 1]

                y_fake_e = self.forward_discriminator(e_hat)
                # y_fake_e.shape = [batch_size, sequence_length, 1]

                G_loss_U = self.__compute_loss_unsupervised(torch.ones_like(y_fake), y_fake)
                G_loss_U_e = self.__compute_loss_unsupervised(torch.ones_like(y_fake_e), y_fake_e)
                loss_unsupervised = G_loss_U + G_loss_U_e

                loss_supervised = self.__compute_loss_supervised(h[:, 1:, :], h_hat_s[:, :-1, :])

                loss_stdmean = self.__compute_loss_stdmean(x, x_hat)

                loss_generator = loss_unsupervised + 100 * torch.sqrt(loss_supervised) + 100 * loss_stdmean

                self.log_correlation_distances(x, x_hat, stage="train")
                self.log_dict({"loss/generator": loss_generator}, on_step=True, on_epoch=True, prog_bar=True)
                self.manual_backward(loss_generator)

                opt_generator.step()
                opt_supervisor.step()

                x_tilde = self.forward_recoverer(h)
                # x_tilde.shape = [batch_size, n_features, sequence_length]

                G_loss_S = self.__compute_loss_supervised(h[:, 1:, :], h_hat_s[:, :-1, :])
                E_loss_T0 = self.__compute_loss_supervised(x, x_tilde)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S

                self.log_dict({"loss/generator-autoencoder": E_loss}, on_step=True, on_epoch=True, prog_bar=True)
                self.manual_backward(E_loss)

                opt_embedder.step()
                opt_recoverer.step()

            h = self.forward_embedder(x).detach()
            # h.shape = [batch_size, sequence_length, hidden_size]

            e_hat = self.forward_generator(batch_size).detach()
            # e_hat.shape = [batch_size, sequence_length, hidden_size]

            h_hat = self.forward_supervisor(e_hat).detach()
            # h_hat.shape = [batch_size, sequence_length, hidden_size]

            y_real = self.forward_discriminator(h)
            # y_real.shape = [batch_size, sequence_length, 1]

            y_fake = self.forward_discriminator(h_hat)
            # y_fake.shape = [batch_size, sequence_length, 1]

            y_fake_e = self.forward_discriminator(e_hat)
            # y_fake_e.shape = [batch_size, sequence_length, 1]

            D_loss_real = self.__compute_loss_unsupervised(torch.ones_like(y_real), y_real)
            D_loss_fake = self.__compute_loss_unsupervised(torch.zeros_like(y_fake), y_fake)
            D_loss_fake_e = self.__compute_loss_unsupervised(torch.zeros_like(y_fake_e), y_fake_e)
            D_loss = D_loss_real + D_loss_fake + D_loss_fake_e

            self.log_dict({"loss/discriminator": D_loss}, on_step=True, on_epoch=True, prog_bar=True)

            if D_loss > self.hparams.discriminator_threshold:
                self.manual_backward(D_loss)
                opt_discriminator.step()

    def __validation_n_test_epoch_end(self, samples: Sequence[Dict]) -> None:

        # Aggregation of the batches
        x, prices, volumes = list(), list(), list()
        for batch in samples:
            x.append(batch["x"])
            if self.hparams.target_feature_price is not None:
                prices.append(batch["prices"])
            if self.hparams.target_feature_volume is not None:
                volumes.append(batch["volumes"])

        # Building the whole real sequences
        x = torch.concatenate(x, dim=2).detach().cpu()
        dict_with_reals = dict(x=x)
        if self.hparams.target_feature_price is not None:
            prices = torch.concatenate(prices, dim=2).detach().cpu()
            dict_with_reals["prices"] = prices
        if self.hparams.target_feature_volume is not None:
            volumes = torch.concatenate(volumes, dim=2).detach().cpu()
            dict_with_reals["volumes"] = volumes

        dict_with_preds = self.predict_autoregressively(prediction_length=x.shape[2])

        # Saving preds and reals in pickle files
        self.save_files(dict_with_reals, dict_with_preds)

        # Retrieving the pred_sequence and the real sequence
        x_hat = dict_with_preds["x_hat"]

        # Logging the correlations metrics
        self.log_correlations(x_hat, "pred")
        self.log_correlations(x, "real")
        self.log_correlation_distances(x, x_hat, "val")

        if self.current_epoch > 0:

            pred_prices = None
            # If there are prices
            if self.hparams.target_feature_price is not None:
                pred_prices = dict_with_preds["pred_prices"].numpy()

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
                pred_volumes = dict_with_preds["pred_volumes"].numpy()

                # Plot volumes
                self.log_plot_timeseries(volumes, pred_volumes, "Volumes")

                # Logging volumes metrics
                self.log_metrics_volume(volumes, pred_volumes)

            # If there are both prices and volumes
            if self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
                sequence_price, pred_sequence_price = (
                    x[: self.hparams.n_stocks],
                    x_hat[: self.hparams.n_stocks],
                )
                sequence_volume, pred_sequence_volume = (
                    x[self.hparams.n_stocks :],
                    x_hat[self.hparams.n_stocks :],
                )

                # Plot stylised fact
                if self.hparams.dataset_type == "multistock":
                    self.log_plot_sf_volume_volatility_correlation(
                        sequence_price, pred_sequence_price, sequence_volume, pred_sequence_volume
                    )

    def predict_autoregressively(self, prediction_length):
        prediction_iterations = math.ceil(prediction_length / self.hparams.sequence_length)

        x_hat = list()
        for i in range(prediction_iterations):
            e_hat = self.forward_generator(batch_size=1)
            h_hat = self.forward_supervisor(e_hat)
            x_hat_ = self.forward_recoverer(h_hat)
            # x_hat_.shape = [1, n_features, sequence_length]
            x_hat_ = x_hat_.squeeze().detach().cpu()
            # x_hat_.shape = [n_features, sequence_length]
            x_hat.append(x_hat_)

        x_hat = torch.concatenate(x_hat, dim=1)
        # x_hat.shape = [n_features, prediction_length]

        return_dict = dict(x_hat=x_hat)

        x_hat = x_hat.numpy()

        x_hat_prices = None
        x_hat_volumes = None

        # If there are prices
        if self.hparams.target_feature_price is not None:
            x_hat_prices = x_hat[: self.hparams.n_stocks, :]
            pred_prices = self.pipeline_price.inverse_transform(x_hat_prices.T).T
            return_dict["pred_prices"] = torch.Tensor(pred_prices)

        # If there are both prices and volumes
        elif self.hparams.target_feature_price is not None and self.hparams.target_feature_volume is not None:
            x_hat_volumes = x_hat[self.hparams.n_stocks :, :]

        # If there are only volumes
        elif self.hparams.target_feature_volume is not None:
            x_hat_volumes = x_hat[: self.hparams.n_stocks, :]

        if self.hparams.target_feature_volume is not None:
            pred_volumes = self.pipeline_volume.inverse_transform(x_hat_volumes.T).T
            return_dict["pred_volumes"] = torch.Tensor(pred_volumes)

        return return_dict

    def __compute_loss_recontruction(self, x, x_tilde):
        return self.mse(x, x_tilde)

    def __compute_loss_supervised(self, h, h_hat):
        return self.mse(h, h_hat)

    def __compute_loss_unsupervised(self, zeros_or_ones, y):
        return self.bcewl(zeros_or_ones, y)

    def __compute_loss_stdmean(self, x, x_hat):
        G_loss_V1 = torch.mean(
            torch.abs(
                torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
            )
        )

        G_loss_V2 = torch.mean(torch.abs((x_hat.mean(dim=0)) - (x.mean(dim=0))))

        return G_loss_V1 + G_loss_V2

    def configure_optimizers(self):
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

        opt_supervisor = hydra.utils.instantiate(
            self.hparams.optimizer_supervisor,
            params=self.recoverer.parameters(),
            _convert_="partial",
        )

        opt_generator = hydra.utils.instantiate(
            self.hparams.optimizer_generator,
            params=self.generator.parameters(),
            _convert_="partial",
        )

        opt_discriminator = hydra.utils.instantiate(
            self.hparams.optimizer_discriminator,
            params=self.generator.parameters(),
            _convert_="partial",
        )

        return [opt_embedder, opt_recoverer, opt_supervisor, opt_generator, opt_discriminator]


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
