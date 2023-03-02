import logging
import warnings
from typing import Dict, List

import dotenv
import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import thesis_gan  # noqa
from thesis_gan.common.utils import complete_configuration

warnings.filterwarnings("ignore", category=DeprecationWarning)

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)
    dotenv.load_dotenv()

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.data.num_workers = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.data.module['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.module, _recursive_=False)

    metadata: Dict = getattr(datamodule, "metadata", None)

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.model.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model.module, _recursive_=False, metadata=metadata)

    # Instantiate the callbacks
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    ckpt_path = cfg.train["checkpoint"]
    if ckpt_path is not None:
        pylogger.info(f"Loading {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        pylogger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig):
    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

    cfg = complete_configuration(cfg)

    run(cfg)


if __name__ == "__main__":
    main()
