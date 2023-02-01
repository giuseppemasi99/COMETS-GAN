import numpy as np
import pandas as pd
import scipy.stats as scs
import torch


def corr(x_batch: torch.Tensor) -> torch.Tensor:
    if len(x_batch.shape) == 2:
        x_batch = x_batch.unsqueeze(0)

    n_features = x_batch.shape[1]
    indices = torch.triu_indices(n_features, n_features, 1)

    correlations = []
    for x in x_batch:
        correlation = torch.corrcoef(x)
        correlation = correlation[indices[0], indices[1]]
        correlations.append(correlation)

    return torch.stack(correlations)


def autocorrelation(series: np.ndarray) -> np.ndarray:
    n = len(series)

    def r(h: float) -> float:
        return ((series[: n - h] - mean) * (series[h:] - mean)).sum() / n / c0

    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n
    x = np.arange(n) + 1
    y = np.array([r(loc) for loc in x])
    return y


def compute_avg_log_returns(x, delta):
    x = x.T
    # x.shape = [sequence_length, n_stocks]
    x = pd.DataFrame(x)
    x = x.rolling(delta).mean().to_numpy().squeeze()
    x = x[::delta][1:]
    return x.T


def compute_avg_volumes(x, delta):
    x = x.T
    # x.shape = [sequence_length, n_stocks]
    x = pd.DataFrame(x)
    x = x.rolling(delta).mean().to_numpy().squeeze()
    x = x[::delta][1:]
    return x.T


# extract all the stats from describe() function
def extract_data_stats(col):
    d_stat = col.describe()
    mu = d_stat["mean"]
    sigma = d_stat["std"]
    rtn_range = np.linspace(d_stat["min"], d_stat["max"], num=1000)
    norm_pdf = scs.norm.pdf(rtn_range, loc=mu, scale=sigma)
    return mu, sigma, rtn_range, norm_pdf
