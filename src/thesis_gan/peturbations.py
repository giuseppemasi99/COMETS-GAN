"""Docstring."""
import pickle
from typing import Dict

import pytorch_lightning as pl

RUN_ID_PRICE = "24uxrxqz"
CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_274.ckpt"

SIGMA_SCALER = 0.7


def run_perburbation_experiment(model: pl.LightningModule, datamodule: pl.LightningDataModule, metadata: Dict):
    """Docstring."""
    val_datasets = datamodule.val_datasets[0]
    x = val_datasets[0]["x"]

    model = model.load_from_checkpoint(CHECKPOINT_PATH, metadata=metadata)

    dict_with_synthetic_perturbed = model.predict_autoregressively(x, prediction_length=3900, sigma_scaler=SIGMA_SCALER)
    with open(
        f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/perturbation_sigmascaler={SIGMA_SCALER}.pickle", "wb"
    ) as f:
        pickle.dump(dict_with_synthetic_perturbed, f, pickle.HIGHEST_PROTOCOL)
