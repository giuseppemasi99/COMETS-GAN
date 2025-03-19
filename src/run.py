import os
import pickle


# Min-Max [-1, 1] also on prices : tanh

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset.datamodule import MyDataModule
from model.pl_model import MyLightningModule


def launch(cfg: DictConfig) -> str:
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

    # logger.log_hyperparams(extract_params(cfg))

    datamodule: MyDataModule = instantiate(cfg.datamodule, _recursive_=False)
    is_prices = cfg.dataset.target_feature_price is not None
    is_volumes = cfg.dataset.target_feature_volume is not None

    os.makedirs(cfg.path_storage, exist_ok=True)
    with open(f'{cfg.path_storage}/pipeline.pkl', 'wb') as f:
        d = dict(pipeline_price=datamodule.pipeline_price, pipeline_volume=datamodule.pipeline_volume)
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    model: MyLightningModule = instantiate(
        cfg.model, stock_names=cfg.dataset.stock_names, 
        is_prices=is_prices, is_volumes=is_volumes, 
        pipeline_price=datamodule.pipeline_price, pipeline_volume=datamodule.pipeline_volume,
        path_storage=cfg.path_storage,
        _recursive_=False
    )

    trainer: Trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule)

    wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig):

    def resolve_tuple(*args):
        return tuple(args)
    OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

    if cfg.trainer.fast_dev_run:
        cfg.logger.mode = 'disabled'

    with open_dict(cfg):
        cfg.trainer.accelerator = 'cuda' if not cfg.trainer.fast_dev_run and torch.cuda.is_available() else 'cpu'
        cfg.trainer.devices = "auto" if torch.cuda.device_count() == 0 else torch.cuda.device_count()

        cfg.datamodule.num_workers = os.cpu_count() if cfg.trainer.accelerator == 'cuda' else 0
        cfg.datamodule.pin_memory = cfg.trainer.accelerator == 'cuda'

    launch(cfg)

if __name__ == "__main__":
    main()
