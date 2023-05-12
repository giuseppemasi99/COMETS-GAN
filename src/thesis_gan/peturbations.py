"""Docstring."""
import pickle
from typing import Dict

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig

# PEP & KO
RUN_ID_PRICE = "24uxrxqz"
CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_274.ckpt"

# PEP & KO & NVDA
RUN_ID_PRICE = "oy1upczq"
CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_192.ckpt"

# PEP & KO, decoder_length = 150
RUN_ID_PRICE = "19oly5ng"
CHECKPOINT_PATH = f"storage/thesis-gan/{RUN_ID_PRICE}/checkpoints/checkpoint_118.ckpt"


SIGMA_SCALER = 0.4


def run_perburbation_experiment(
    cfg: DictConfig, model: pl.LightningModule, datamodule: pl.LightningDataModule, metadata: Dict, sampling_seed: int
):
    """Docstring."""
    val_datasets = datamodule.val_datasets[0]
    x = val_datasets[0]["x"]

    model = model.load_from_checkpoint(CHECKPOINT_PATH, metadata=metadata)

    KOret_WoPer = None
    if SIGMA_SCALER is not None:
        fpath = f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/shock/perturbation_seed=10_sigmascaler=None.pickle"
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        returnsWoPer = d["x_hat"].numpy()
        KOret_WoPer, _ = returnsWoPer

    dict_with_synthetic_perturbed = model.predict_autoregressively(
        x, prediction_length=7000, sigma_scaler=SIGMA_SCALER, KOret_WoPer=KOret_WoPer
    )

    pred_prices = dict_with_synthetic_perturbed["pred_prices"]
    pred_corr = np.corrcoef(pred_prices)
    print("\n", pred_corr, "\n")

    with open(
        f"storage/thesis-gan/{RUN_ID_PRICE}/perturbations/shock/perturbation"
        f"_seed={sampling_seed}"
        f"_sigmascaler={SIGMA_SCALER}"
        # "_downup"
        "_updown" f".pickle",
        "wb",
    ) as f:
        pickle.dump(dict_with_synthetic_perturbed, f, pickle.HIGHEST_PROTOCOL)
