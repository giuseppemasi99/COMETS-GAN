"""Docstring."""
import pickle
from typing import Dict

import pytorch_lightning as pl
from omegaconf import DictConfig

# PEP & KO
RUN_ID_PRICE = "24uxrxqz"
CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_274.ckpt"

# PEP & KO & NVDA
# RUN_ID_PRICE = "oy1upczq"
# CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_192.ckpt"

# PEP & KO, decoder_length = 50
# RUN_ID_PRICE = "19oly5ng"
# CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_118.ckpt"

SIGMA_SCALER = 0.4
LINEAR = False


def run_perburbation_experiment(
    cfg: DictConfig, model: pl.LightningModule, datamodule: pl.LightningDataModule, metadata: Dict, sampling_seed: int
):
    """Docstring."""
    val_datasets = datamodule.val_datasets[0]
    x = val_datasets[0]["x"]

    model = model.load_from_checkpoint(CHECKPOINT_PATH, metadata=metadata)

    dict_with_synthetic_perturbed = model.predict_autoregressively(
        x, prediction_length=3900 * 2, sigma_scaler=SIGMA_SCALER, linear=LINEAR
    )

    with open(
        f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/"
        f"22_06/"
        f"perturbation"
        f"_seed={sampling_seed}"
        f"_sigmascaler={SIGMA_SCALER}"
        f"_linear={LINEAR}"
        f".pickle",
        "wb",
    ) as f:
        pickle.dump(dict_with_synthetic_perturbed, f, pickle.HIGHEST_PROTOCOL)
