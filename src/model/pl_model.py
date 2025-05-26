import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from dataset.preprocessing import Pipeline
from model.csdi.csdi import CSDI
from model.diffusion import Diffusion
from model.gan.critic import CNNCritic


class MyPLModel(LightningModule):

    def __init__(
        self, model: CSDI, diffusion: Diffusion, pipeline: Pipeline,
        guidance_scale: float, critic_ckpt_path: str, critic: CNNCritic,
        batch_size: int, lr: float, 
        path_storage: str, n_samples_evaluation: int, 
        seq_len: int, n_features: int, 
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.path_storage = path_storage
        self.n_samples_evaluation = n_samples_evaluation
        self.seq_len = seq_len
        self.n_features = n_features

        self.diffusion = diffusion
        self.model = model

        self.critic_ckpt_path = critic_ckpt_path
        self.guidance_scale = guidance_scale
        self.critic = critic

        self.pipeline = pipeline

        self.loss_reconstruction = nn.MSELoss()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_0 = batch["x_0"]  # [B, seq_len, n_features]

        t = torch.randint(0, self.diffusion.diffusion_timesteps, size=(self.batch_size,), device=self.device, dtype=torch.long)  # [B]

        x_t, noise_real = self.diffusion.forward_diffusion(x_0, t)
        # x_t.shape = noise_real.shape = [B, seq_len, n_features]
        
        noise_pred = self.model(x_t, t)  # [B, seq_len, n_features]

        return self.loss_reconstruction(noise_pred, noise_real)

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        loss = self(batch)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor]) -> None:
        loss = self(batch)
        self.log('val/loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        samples = np.empty((0, self.seq_len, self.n_features))
        n = self.n_samples_evaluation
        while n > 0:
            x_t = torch.randn((min(self.batch_size, n), self.seq_len, self.n_features), device=self.device)
            samples = np.append(samples, self.predict_step(x_t), axis=0)
            n = max(n-self.batch_size, 0)
        
        path = f'{self.path_storage}/epoch={self.current_epoch}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/samples.npy', 'wb') as f:
            np.save(f, samples)

    def predict_step(self, x_t: Tensor) -> np.ndarray:
        for i in reversed(range(self.diffusion.diffusion_timesteps)):
            t = torch.full((self.batch_size,), i, device=self.device, dtype=torch.long)  # [B]
            noise_pred = self.model(x_t, t)  # [B, seq_len, n_features]
            if self.guidance_scale > 0:
                x_t.requires_grad_()
                with torch.enable_grad():
                    critic_score: Tensor = self.critic(x_t).mean()
                    critic_score.backward()
                noise_pred += self.guidance_scale * x_t.grad
            x_t = self.diffusion.backward_diffusion(x_t, noise_pred, t, i)  # [B, seq_len, n_features]
        samples = x_t.squeeze().detach().cpu().numpy()  # [B, seq_len, n_features]
        samples = np.asarray([self.pipeline.inverse_transform(s) for s in samples])  # [B, seq_len, n_features]
        return samples

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        return {'optimizer': optimizer}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        state_dict_critic: Dict[str, Tensor] = torch.load(self.critic_ckpt_path, weights_only=False)['state_dict']
        for key in state_dict_critic:
            if 'discriminator' in key:
                checkpoint['state_dict'][key.replace('discriminator', 'critic')] = state_dict_critic[key]
        return super().on_load_checkpoint(checkpoint)