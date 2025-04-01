import os
import numpy as np
import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict
from utils.save import save
from data.datamodule import MyDataModule
from model.pl_model import MyPLModel
from utils.extract_params import extract_params


def launch(cfg: DictConfig) -> None:
    print(f'Seed={cfg.seed}')
    seed_everything(cfg.seed, workers=True)

    logger: WandbLogger = instantiate(cfg.logger)
    if not cfg.trainer.fast_dev_run and cfg.logger.mode != 'disabled':
        artifact = wandb.Artifact('configs', type='dataset')
        artifact.add_dir('src/conf', 'configs')
        logger.experiment.log_artifact(artifact)

    with open_dict(cfg):
        cfg.path_storage = f"storage/{'fdr' if cfg.trainer.fast_dev_run else logger.experiment.id}/"
        cfg.path_checkpoint = cfg.path_storage + "/checkpoints"

    logger.log_hyperparams(extract_params(cfg))
    
    datamodule: MyDataModule = instantiate(cfg.datamodule, _recursive_=False)

    pl_model: MyPLModel = instantiate(cfg.pl_model, pipeline=datamodule.pipeline)

    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    if cfg.ckpt_path:
        synthetic = np.concat(trainer.predict(
            model=pl_model, datamodule=datamodule, return_predictions=True, ckpt_path=cfg.ckpt_path
        ), axis=0)
        epoch = int(cfg.ckpt_path.split('epoch=')[1].split('-')[0])
        inference_data_path = cfg.ckpt_path[:cfg.ckpt_path.index('checkpoints')] + 'inference_data/' 
        inference_data_path += f'guidance_scale={cfg.pl_model.guidance_scale}/'
        if cfg.pl_model.guidance_scale > 0:
            inference_data_path += f'critic_epoch={cfg.pl_model.critic_ckpt_path.split('epoch=')[1].split('.')[0]}/'
        save(inference_data_path, f'synthetics_epoch={epoch}_seed={cfg.seed}', synthetic)

    else:
        trainer.fit(model=pl_model, datamodule=datamodule)
        if cfg.trainer.fast_dev_run:
            trainer.predict(model=pl_model, datamodule=datamodule)

    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    if cfg.trainer.fast_dev_run or cfg.ckpt_path:
        cfg.logger.mode = 'disabled'

    with open_dict(cfg):
        cfg.trainer.accelerator = 'cuda' if not cfg.trainer.fast_dev_run and torch.cuda.is_available() else 'cpu'
        cfg.trainer.devices = "auto" if torch.cuda.device_count() == 0 else torch.cuda.device_count()
        cfg.datamodule.num_workers = os.cpu_count() if cfg.trainer.accelerator == 'cuda' else 0
        cfg.datamodule.pin_memory = cfg.trainer.accelerator == 'cuda'
        cfg.pl_model.n_features = len(cfg.dataset.feature_names)
        cfg.datamodule.dataset_predict.n_features = len(cfg.dataset.feature_names)

    launch(cfg)


if __name__ == "__main__":
    main()
