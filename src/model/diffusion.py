from typing import Tuple

import torch
from torch import Tensor


class Diffusion:

    def __init__(self, diffusion_timesteps: int, beta_scheduler_type: str, device: str) -> None:
        self.device = device

        self.diffusion_timesteps = diffusion_timesteps
        self.beta_scheduler_type = beta_scheduler_type

        self.betas: Tensor = self.__beta_schedule(self.diffusion_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Forward Diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Backward Diffusion
        self.coeff1 = 1 / torch.sqrt(self.alphas)
        self.coeff2 = (1 - self.alphas) / (1 - self.alphas_cumprod) ** 0.5
        alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sigma = torch.sqrt((1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod) * self.betas)

    def forward_diffusion(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.randn_like(x_0)
        # x_0.shape = noise.shape = [B, seq_len, n_features]

        sqrt_alphas_cumprod = self.extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod = self.extract(self.sqrt_one_minus_alphas_cumprod, t)

        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise  # [B, seq_len, n_features]

        return x_t, noise

    def backward_diffusion(self, x_t: Tensor, noise_pred: Tensor, t: Tensor, timestep: int) -> Tensor:
        # x_t.shape = noise_pred.shape = [B, seq_len, n_features]
        coeff1 = self.extract(self.coeff1, t)
        coeff2 = self.extract(self.coeff2, t)
        
        x_tm1 = coeff1 * (x_t - coeff2 * noise_pred)
        
        if timestep > 0:
            sigma = self.extract(self.sigma, t)
            x_tm1 += sigma * torch.randn_like(x_tm1)

        return x_tm1  # [B, seq_len, n_features]

    @staticmethod
    def extract(tensor: Tensor, t: Tensor) -> Tensor:
        # tensor.shape = [diffusion_timesteps,]
        # t.shape = [B,]
        return tensor.gather(-1, t).reshape(t.shape[0], *((1,) * 2))

    def __beta_schedule(self, diffusion_timesteps) -> Tensor:
        if self.beta_scheduler_type == 'Linear':
            return self.__linear_beta_schedule(diffusion_timesteps)
        elif self.beta_scheduler_type == 'Quadratic':
            return self.__quadratic_beta_schedule(diffusion_timesteps)
        elif self.beta_scheduler_type == 'Cosine':
            return self.__cosine_beta_schedule(diffusion_timesteps)
        else:
            return self.__sigmoid_beta_schedule(diffusion_timesteps)

    def __linear_beta_schedule(self, diffusion_timesteps, start: float = .0001, end: float = .02) -> Tensor:
        return torch.linspace(start, end, diffusion_timesteps, device=self.device)

    def __quadratic_beta_schedule(self, diffusion_timesteps, start: float = 1e-06, end: float = .5) -> Tensor:
        return torch.linspace(start ** .5, end ** .5, diffusion_timesteps, device=self.device) ** 2

    def __cosine_beta_schedule(self, diffusion_timesteps, s: float = .008) -> Tensor:
        steps = diffusion_timesteps + 1
        x = torch.linspace(0, diffusion_timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / diffusion_timesteps) + s) / (1 + s) * torch.pi * .5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, .0001, .9999)

    def __sigmoid_beta_schedule(self, diffusion_timesteps, start: float = .0001, end: float = .02) -> Tensor:
        betas = torch.linspace(-6, 6, diffusion_timesteps, device=self.device)
        return torch.sigmoid(betas) * (end - start) + start
