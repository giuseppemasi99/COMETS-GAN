from typing import Any, Dict

from omegaconf import DictConfig


def extract_params(cfg: DictConfig) -> Dict[str, Any]:
    return dict(
        seed=cfg.seed,
        dataset=cfg.dataset.name,
        feature_names=cfg.dataset.feature_names,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        diffusion_timesteps=cfg.pl_model.diffusion.diffusion_timesteps,
        max_epochs=cfg.trainer.max_epochs
    )
