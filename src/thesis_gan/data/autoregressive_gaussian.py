from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nn_core.nn_types import Split

from thesis_gan.data.pipeline import Pipeline


def ar_gaussian_generation(
    n_samples: int, seq_len: int, n_features: int, phi: float, sigma: float, seed: Optional[int] = None
) -> np.ndarray:
    # Set seed for reproducibility
    np.random.seed(seed)

    mean_val = np.zeros([n_features])
    cov_val = np.ones([n_features, n_features]) * sigma
    np.fill_diagonal(cov_val, 1)

    # Generate autoregressive gaussian data
    data = []
    for _ in tqdm(range(n_samples), desc="Dataset generation"):
        temp = np.zeros([seq_len, n_features])
        for k in range(seq_len):
            # Starting feature
            if k == 0:
                temp[k, :] = np.random.multivariate_normal(mean_val, cov_val, 1)

            # AR(1) Generation
            else:
                temp[k, :] = phi * temp[k - 1, :] + (1 - phi) * 1 * np.random.multivariate_normal(mean_val, cov_val, 1)

        data.append(temp)

    return np.array(data)


class ARGaussianDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        encoder_length: int,
        decoder_length: int,
        n_features: int,
        phi: float,
        sigma: float,
        split: Split,
        data_pipeline: Pipeline,
        seed: Optional[int] = None,
    ) -> None:
        super(ARGaussianDataset, self).__init__()
        seq_len = encoder_length + decoder_length
        data = ar_gaussian_generation(n_samples, seq_len, n_features, phi, sigma, seed)
        self.data = torch.as_tensor(data, dtype=torch.float).permute(0, 2, 1)
        self.n_samples = n_samples
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_features = n_features
        self.phi = phi
        self.sigma = sigma
        self.data_pipeline = data_pipeline
        self.split = split

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = self.data[index][..., : self.encoder_length]
        y = self.data[index][..., self.encoder_length : self.encoder_length + self.decoder_length]
        return {"x": x, "y": y}

    def __repr__(self) -> str:
        props = (
            f"{self.split=}, {self.n_samples=}, {self.encoder_length=}, {self.decoder_length=},"
            f"{self.n_features=}, {self.phi=}, {self.sigma=}"
        )
        props = props.replace("self.", "")
        return f"ARGaussianDataset({props})"
