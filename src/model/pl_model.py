import os
import pickle
from itertools import combinations
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        self, encoder_length: int, decoder_length: int, generation_length: int,
        generator: DictConfig, discriminator: DictConfig, n_critic: int, 
        stock_names: str, pipeline_price: Pipeline, pipeline_volume: Pipeline,
        path_storage: str
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.path_storage = path_storage

        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

        self.pipeline_price = pipeline_price
        self.pipeline_volume = pipeline_volume

        self.n_stocks = len(stock_names)
        n_features = self.n_stocks*2

        self.stock_names = stock_names
        self.feature_names = [f'{s}_price' for s in stock_names] + [f'{s}_volume' for s in stock_names]

        self.generator: TCNGenerator = instantiate(
            generator, n_features=n_features, n_stocks=self.n_stocks,
        )

        self.discriminator: CNNDiscriminator = instantiate(
            discriminator, n_features=n_features,
        )

        self.λ = 0
        self.n_critic = n_critic

        self.mse = nn.MSELoss(reduction='none')

        self.generation_length = generation_length

    def forward(self, x: torch.Tensor, noise: torch.Tensor, t_past: torch.Tensor) -> torch.Tensor:
        out = self.generator(x, noise, t_past)
        return out

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()
        opt_g: Optimizer
        opt_d: Optimizer

        x, y_real = batch["x"], batch["y"]
        # x.shape [B, n_features, encoder_length]
        # y_real.shape [B, n_features, decoder_length]

        t_past, t_fut = batch["t_past"], batch["t_fut"]
        # t_past.shape [B, encoder_length]
        # t_fut.shape [B, decoder_length]

        noise = torch.randn((x.shape[0], 1, self.encoder_length), device=self.device)
        # noise.shape = [B, 1, encoder_length]

        # Train discriminator
        if batch_idx == 0 or batch_idx % self.n_critic != 0:
            for p in self.discriminator.parameters():
                p.requires_grad = True
            for p in self.generator.parameters():
                p.requires_grad = False

            y_pred = self(x, noise, t_past)

            real_validity = self.discriminator(x, y_real, t_past, t_fut)
            fake_validity = self.discriminator(x, y_pred, t_past, t_fut)
            
            opt_d.zero_grad()
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity)
            d_loss += self.λ * self.calculate_gradient_penalty(x, y_real, y_pred, t_past, t_fut).mean()
            self.log("loss/discriminator", d_loss, prog_bar=True)
            d_loss.backward()
            opt_d.step()

        # Train generator
        else:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.generator.parameters():
                p.requires_grad = True

            opt_g.zero_grad()
            y_pred = self(x, noise, t_past)
            g_loss = -torch.mean(self.discriminator(x, y_pred, t_past, t_fut))
            self.log("loss/generator", g_loss, prog_bar=True)
            if self.n_stocks > 1:
                self.log_corr_dist(y_real, y_pred)
            g_loss.backward()
            opt_g.step()

    def calculate_gradient_penalty(
        self, x: torch.Tensor, y_real: torch.Tensor, y_pred: torch.Tensor,
        t_past: torch.Tensor, t_fut: torch.Tensor
    ) -> torch.Tensor:
        ɛ = torch.rand((len(y_real), 1, 1), device=self.device)
        interpolated = ɛ * y_real + (1 - ɛ) * y_pred

        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        prob_interpolated = self.discriminator(x, interpolated, t_past, t_fut)

        grad_outputs = torch.ones_like(prob_interpolated).to(self.device)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated, inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0].flatten(1)
        
        grad_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return grad_penalty

    def log_corr_dist(self, y_real: torch.Tensor, y_pred: torch.Tensor) -> None:
        corr_real, corr_pred = corr(y_real[:, :self.n_stocks]), corr(y_pred[:, :self.n_stocks])
        metric_names = [f"corr_dist/{'-'.join(x)}" for x in combinations(self.feature_names[:self.n_stocks], 2)]
        corr_distances = self.mse(corr_real, corr_pred).mean(dim=0)
        d = {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)}
        self.log_dict(d, prog_bar=False)
        self.log('corr_dist/mean', corr_distances.mean(), prog_bar=True)

    def predict_autoregressively(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat = x[:, :, :self.encoder_length]

        # prediction_iterations = math.ceil(self.generation_length / self.decoder_length)
        for i in range(self.encoder_length, self.generation_length+self.encoder_length, self.decoder_length):
            noise = torch.randn(len(x_hat), 1, self.encoder_length, device=self.device)
            o = self(
                x_hat[:, :, -self.encoder_length:], noise, 
                t[:, i-self.encoder_length: i]
            )
            x_hat = torch.cat((x_hat, o), dim=2)

        x_hat = x_hat.detach().cpu().numpy()[:, :, :self.generation_length+self.encoder_length]
        x = x.detach().cpu().numpy()[:, :, :self.generation_length+self.encoder_length]

        x_hat_price = np.asarray([self.pipeline_price.inverse_transform(x_hat_[:self.n_stocks].T).T for x_hat_ in x_hat]).squeeze()
        x_hat_volume = np.asarray([self.pipeline_volume.inverse_transform(x_hat_[self.n_stocks:].T).T for x_hat_ in x_hat]).squeeze()

        x_price = np.asarray([self.pipeline_price.inverse_transform(x_[:self.n_stocks].T).T for x_ in x]).squeeze()
        x_volume = np.asarray([self.pipeline_volume.inverse_transform(x_[self.n_stocks:].T).T for x_ in x]).squeeze()

        return dict(x_hat_price=x_hat_price, x_price=x_price, x_hat_volume=x_hat_volume, x_volume=x_volume)

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> None:
        x = batch["x"]
        t = batch["t"]
        # x.shape [1, time_series_length]
        # t.shape [1, time_series_length]

        d = self.predict_autoregressively(x, t)

        path = f'{self.path_storage}/synthetic/epoch={self.current_epoch}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/sample.pkl', 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        fig, axes = plt.subplots(2, 4, figsize=(10, 4))
        label = True
        for ax, r, s, f in zip(axes[0], d['x_price'], d['x_hat_price'], self.stock_names):
            ax.plot(r, label='Real' if label else None)
            ax.plot(s, label='Synthetic' if label else None)
            label = False
            ax.set_title(f)
        for ax, r, s in zip(axes[1], d['x_volume'], d['x_hat_volume']):
            ax.plot(r, alpha=.5)   
            ax.plot(s, alpha=.5)
        fig.legend()
        fig.tight_layout()

        path = f'{self.path_storage}/plots'
        os.makedirs(path, exist_ok=True)
        fig.savefig(f'{path}/epoch={self.current_epoch}.png')

        if self.current_epoch % 30 == 0:
            title_wandb = f'samples/Epoch:{self.current_epoch}'
            self.logger.experiment.log({title_wandb: wandb.Image(fig)})
        plt.close(fig)

    def predict_step(self, batch: Dict[str, torch.Tensor]):
        x = batch["x"]
        t = batch["t"]
        d = self.predict_autoregressively(x, t)
        d['y_price'], d['y_volume'] = batch['y_price'].detach().cpu().numpy(), batch['y_volume'].detach().cpu().numpy()
        return d

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=.0001, betas=(0., .9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=.0001, betas=(0., .9))
        return opt_g, opt_d
