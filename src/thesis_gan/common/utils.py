import numpy as np
import pandas as pd
import scipy.stats as scs
import torch
from omegaconf import omegaconf, open_dict


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


def complete_configuration(cfg: omegaconf.DictConfig):
    dataset_type = cfg.data.dataset_type
    if dataset_type == "multistock" or dataset_type == "DowJones":
        stock_names = cfg.data.stock_names
        target_feature_price = cfg.data.target_feature_price
        target_feature_volume = cfg.data.target_feature_volume

        n_stocks = len(cfg.data.stock_names)
        feature_names = list()
        if target_feature_price is not None:
            feature_names.extend([stock_name + "_" + target_feature_price for stock_name in stock_names])
        if target_feature_volume is not None:
            feature_names.extend([stock_name + "_" + target_feature_volume for stock_name in stock_names])
        n_features = len(feature_names)

        cfg.model.module = save_in_cfg(
            cfg.model.module,
            stock_names=stock_names,
            target_feature_price=target_feature_price,
            target_feature_volume=target_feature_volume,
            n_stocks=n_stocks,
            feature_names=feature_names,
            n_features=n_features,
        )

    else:
        n_features = cfg.data.n_features
        cfg.model.module = save_in_cfg(cfg.model.module, n_features=n_features)

    model_type = cfg.model.model_type
    if model_type == "timegan":
        sequence_length = cfg.model.sequence_length
        cfg.data.module.datasets.train = save_in_cfg(cfg.data.module.datasets.train, sequence_length=sequence_length)
    else:
        encoder_length = cfg.data.encoder_length
        decoder_length = cfg.data.decoder_length
        cfg.data.module.datasets.train = save_in_cfg(
            cfg.data.module.datasets.train, encoder_length=encoder_length, decoder_length=decoder_length
        )

    # TODO: add sequence_length (if timegan), encoder_length&decoder_length (if conv)
    #  also to val and test set for sines and gaussian

    return cfg


def save_in_cfg(sub_cfg, **args):
    with open_dict(sub_cfg):
        for k, v in args.items():
            sub_cfg[k] = v
    return sub_cfg
