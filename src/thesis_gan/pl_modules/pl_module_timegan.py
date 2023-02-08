import logging
import math
from typing import Dict, Optional, Sequence

import hydra
import numpy as np
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

        self.generator = hydra.utils.instantiate(
            self.hparams.generator,
            noise_dim=self.hparams.noise_dim,
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
        # batch.keys() = ['x']

        opt_embedder, opt_recoverer, opt_generator, opt_discriminator = self.optimizers()

        x = batch["x"]
        # x.shape = [batch_size, n_features, sequence_length]

        batch_size = x.shape[0]

        # Training auto-encoder only
        if self.current_epoch < self.hparams.n_epochs_training_only_autoencoder:
            opt_embedder.zero_grad()
            opt_recoverer.zero_grad()

            x_tilde = self.forward_autoencoder(x)
            # x_tilde.shape = [batch_size, n_features, sequence_length]

            loss_reconstruction = 10 * torch.sqrt(self.__compute_loss_recontruction(x, x_tilde))

            self.log_dict({"loss/autoencoder": loss_reconstruction}, on_step=True, on_epoch=True, prog_bar=True)
            self.manual_backward(loss_reconstruction)

            opt_embedder.step()
            opt_recoverer.step()

        # Joint training
        else:
            for _ in range(2):
                opt_generator.zero_grad()
                opt_embedder.zero_grad()
                opt_recoverer.zero_grad()

                h = self.forward_embedder(x)
                # h.shape = [batch_size, sequence_length, hidden_size]

                h_hat = self.forward_generator(batch_size)
                # e_hat.shape = [batch_size, sequence_length, hidden_size]

                x_hat = self.forward_recoverer(h_hat)
                # x_hat.shape = [batch_size, n_features, sequence_length]

                y_real = self.forward_discriminator(h)
                # y_fake.shape = [batch_size, sequence_length, 1]

                y_fake = self.forward_discriminator(h_hat)
                # y_fake_e.shape = [batch_size, sequence_length, 1]

                loss_unsupervised = self.__compute_loss_unsupervised(y_real, y_fake)
                loss_supervised = self.__compute_loss_supervised(h, h_hat)
                loss_stdmean = self.__compute_loss_stdmean(x, x_hat)

                loss_generator = loss_unsupervised + 100 * torch.sqrt(loss_supervised) + 100 * loss_stdmean

                self.log_correlation_distances(x, x_hat, stage="train")
                self.log_dict(
                    {"loss/generator": torch.sqrt(loss_generator)}, on_step=True, on_epoch=True, prog_bar=True
                )
                self.manual_backward(loss_generator, retain_graph=True)

                opt_generator.step()

                x_tilde = self.forward_recoverer(h)
                # x_tilde.shape = [batch_size, n_features, sequence_length]

                loss_reconstruction = 10 * torch.sqrt(self.__compute_loss_recontruction(x, x_tilde))

                self.log_dict(
                    {"loss/joint-autoencoder": loss_reconstruction}, on_step=True, on_epoch=True, prog_bar=True
                )
                self.manual_backward(loss_reconstruction, retain_graph=False)

                opt_embedder.step()
                opt_recoverer.step()

            opt_discriminator.zero_grad()

            h = self.forward_embedder(x).detach()
            # h.shape = [batch_size, sequence_length, hidden_size]

            h_hat = self.forward_generator(batch_size).detach()
            # e_hat.shape = [batch_size, sequence_length, hidden_size]

            y_real = self.forward_discriminator(h)
            # y_real.shape = [batch_size, sequence_length, 1]

            y_fake = self.forward_discriminator(h_hat)
            # y_fake.shape = [batch_size, sequence_length, 1]

            loss_discriminator = self.__compute_loss_unsupervised(
                torch.concatenate((torch.ones_like(y_real), torch.zeros_like(y_fake)), dim=2),
                torch.concatenate((y_real, y_fake), dim=2),
            )

            self.log_dict({"loss/discriminator": loss_discriminator}, on_step=True, on_epoch=True, prog_bar=True)

            if loss_discriminator > self.hparams.discriminator_threshold:
                self.manual_backward(loss_discriminator)
                opt_discriminator.step()

    def validation_n_test_epoch_end(self, samples: Sequence[Dict[str, torch.Tensor]]) -> None:
        if self.current_epoch >= self.hparams.n_epochs_training_only_autoencoder:

            # Aggregation of the batches
            dict_with_reals: Dict[str, torch.Tensor] = self.aggregate_from_batches(samples)

            # Autoregressive prediction
            sequence_length = dict_with_reals["x"].shape[1]
            dict_with_preds = self.predict_autoregressively(prediction_length=sequence_length)

            self.continue_validation_n_test_epoch_end(dict_with_reals, dict_with_preds)

    def predict_autoregressively(self, prediction_length):
        prediction_iterations = math.ceil(prediction_length / self.hparams.sequence_length)

        x_hat = list()
        for i in range(prediction_iterations):
            h_hat = self.forward_generator(batch_size=1)
            x_hat_ = self.forward_recoverer(h_hat)
            # x_hat_.shape = [1, n_features, sequence_length]
            x_hat_ = x_hat_.squeeze().detach().cpu()
            # x_hat_.shape = [n_features, sequence_length]
            x_hat.append(x_hat_)

        x_hat = torch.concatenate(x_hat, dim=1)
        # x_hat.shape = [n_features, prediction_iterations*sequence_length]

        x_hat = x_hat[:, :prediction_length]
        # x_hat.shape = [n_features, prediction_length]

        return self.unpack(x_hat)

    def __compute_loss_recontruction(self, x, x_tilde):
        return self.mse(x, x_tilde)

    def __compute_loss_supervised(self, h, h_hat):
        return self.mse(h[:, 1:, :], h_hat[:, :-1, :])

    def __compute_loss_unsupervised(self, real, pred):
        return self.bcewl(real, pred)

    def __compute_loss_stdmean(self, x, x_hat):
        G_loss_V1 = torch.mean(
            torch.abs(
                torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
            )
        )

        G_loss_V2 = torch.mean(torch.abs((x_hat.mean(dim=0)) - (x.mean(dim=0))))

        return G_loss_V1 + G_loss_V2

    def on_train_start(self) -> None:
        self.log_dict({"loss/generator": np.NaN})

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

        return [opt_embedder, opt_recoverer, opt_generator, opt_discriminator]


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
