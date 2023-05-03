"""Docstring."""
import pickle

import pytorch_lightning as pl

from thesis_gan.pl_modules.pl_module_conv import MyLightningModule

RUN_ID_PRICE = "3m9c18s6"


def run_perburbation_experiment(
    trainer: pl.Trainer, model: MyLightningModule, datamodule: pl.LightningDataModule, metadata, ckpt_path
):
    """Docstring."""
    model = model.load_from_checkpoint("storage/thesis-gan/3m9c18s6/checkpoints/checkpoint_70.ckpt", metadata=metadata)

    val_datasets = datamodule.val_datasets[0]
    x = val_datasets[0]["x"]

    # dict_with_synthetic = model.predict_autoregressively(x, prediction_length=3900)
    # with open(f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/no_perturbation.pickle", 'wb') as f:
    #     pickle.dump(dict_with_synthetic, f, pickle.HIGHEST_PROTOCOL)

    dict_with_synthetic_perturbed = model.predict_autoregressively(x, prediction_length=3900, add_perturbation=True)
    with open(f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/perturbation.pickle", "wb") as f:
        pickle.dump(dict_with_synthetic_perturbed, f, pickle.HIGHEST_PROTOCOL)
