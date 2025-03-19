import math
import os
import pickle
from itertools import combinations
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer

import wandb
from common.utils import corr
from dataset.pipeline import Pipeline
from model.modules.discriminator.cnn import CNNDiscriminator
from model.modules.generator.tcn import TCNGenerator


class MyLightningModule(LightningModule):

    def __init__(
        self, encoder_length: int, decoder_length: int, 
        generator: DictConfig, discriminator: DictConfig, n_critic: int, 
        stock_names: str, is_prices: bool, is_volumes: bool,
        pipeline_price: Pipeline, pipeline_volume: Pipeline,
        path_storage: str
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.path_storage = path_storage

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

        self.pipeline_price = pipeline_price
        self.pipeline_volume = pipeline_volume

        self.is_prices = is_prices
        self.is_volumes = is_volumes
        self.n_stocks = len(stock_names)
        n_features = self.n_stocks*2 if is_prices and is_volumes else self.n_stocks

        self.stock_names = stock_names
        self.feature_names = list()
        if self.is_prices:
            self.feature_names.extend([f'{s}_price' for s in stock_names])
        if self.is_volumes:
            self.feature_names.extend([f'{s}_volume' for s in stock_names])

        self.generator: TCNGenerator = instantiate(
            generator, n_features=n_features, n_stocks=self.n_stocks,
            is_prices=is_prices, is_volumes=is_volumes,
        )

        self.discriminator: CNNDiscriminator = instantiate(
            discriminator, n_features=n_features,
        )

        self.n_critic = n_critic

        self.mse = nn.MSELoss(reduction='none')

        self.generation_length = 3000

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, n_features, encoder_length]
        out = self.generator(x, noise)
        # out.shape = [batch_size, n_features, decoder_length]
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        opt_g: Optimizer
        opt_d: Optimizer

        x, y_real = batch["x"], batch["y"]
        # x.shape [B, n_features, encoder_length]
        # y_real.shape [B, n_features, decoder_length]

        noise = torch.randn((x.shape[0], 1, self.encoder_length), device=self.device)
        # noise.shape = [B, 1, encoder_length]

        # Train discriminator
        if batch_idx > 0 and batch_idx % self.n_critic == 0:
            y_pred = self(x, noise)
            real_validity = self.discriminator(x, y_real)
            fake_validity = self.discriminator(x, y_pred)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log("loss/discriminator", d_loss, prog_bar=True)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        # Train generator
        else:
            y_pred = self(x, noise)
            g_loss = -torch.mean(self.discriminator(x, y_pred))
            self.log("loss/generator", g_loss, prog_bar=True)
            if self.n_stocks > 1:
                self.log_corr_dist(y_real, y_pred)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

    def log_corr_dist(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> None:
        corr_real, corr_pred = corr(y_real), corr(y_pred)
        metric_names = [f"corr_dist/{'-'.join(x)}" for x in combinations(self.feature_names, 2)]
        corr_distances = self.mse(corr_real, corr_pred).mean(dim=0)
        d = {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)}
        self.log_dict(d, prog_bar=False)
        self.log('corr_dist/mean', corr_distances.mean(), prog_bar=True)

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> None:
        x = batch['x']

        x_hat = x[:, :, :self.encoder_length]

        prediction_iterations = math.ceil(self.generation_length / self.decoder_length)

        for _ in range(prediction_iterations):
            noise = torch.randn(1, 1, self.encoder_length, device=self.device)
            o = self(x_hat[:, :, -self.encoder_length:], noise)
            x_hat = torch.cat((x_hat, o), dim=2)
        
        x_hat = x_hat.squeeze().detach().cpu().numpy()
        x = x.squeeze().detach().cpu().numpy()[:, :x_hat.shape[-1]]

        x_hat_price = self.pipeline_price.inverse_transform(x_hat[:self.n_stocks].T).T
        x_price = self.pipeline_price.inverse_transform(x[:self.n_stocks].T).T
        x_hat_volume = self.pipeline_volume.inverse_transform(x_hat[self.n_stocks:].T).T
        x_volume = self.pipeline_volume.inverse_transform(x[self.n_stocks:].T).T

        path = f'{self.path_storage}/synthetic/epoch={self.current_epoch}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/sample.pkl', 'wb') as f:
            d = dict(x_hat_price=x_hat_price, x_price=x_price, x_hat_volume=x_hat_volume, x_volume=x_volume)
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        def make_plot(x, x_hat, type='price'):
            fig, axes = plt.subplots(2, 2, figsize=(7, 5))
            axes = axes.ravel()
            label = True
            for ax, r, s, f in zip(axes, x, x_hat, self.stock_names):
                ax.plot(r, label='Real' if label else None, alpha=.5 if type ==' volume' else 1)
                ax.plot(s, label='Synthetic' if label else None, alpha=.5 if type ==' volume' else 1)
                label = False
                ax.set_title(f)
            fig.legend()
            fig.tight_layout()
            plt.close(fig)
            return fig

        fig_price = make_plot(x_price, x_hat_price)
        fig_volume = make_plot(x_volume, x_hat_volume, type='volume')

        title_wandb = f'prices/Epoch:{self.current_epoch}'
        self.logger.experiment.log({title_wandb: wandb.Image(fig_price)})
        title_wandb = f'volumes/Epoch:{self.current_epoch}'
        self.logger.experiment.log({title_wandb: wandb.Image(fig_volume)})

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=3e-4)
        return opt_g, opt_d
