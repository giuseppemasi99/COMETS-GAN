from itertools import combinations
from typing import Dict, Sequence

import torch
import torch.nn as nn

from thesis_gan.common.utils import corr


def get_correlations_dict(y_realOpred: torch.Tensor, realOpred: str, feature_names: Sequence) -> Dict[str, int]:
    correlations = corr(y_realOpred).squeeze()

    metric_names = [f"{realOpred}_correlation/{'-'.join(x)}" for x in combinations(feature_names, 2)]

    d = {metric: correlation.item() for metric, correlation in zip(metric_names, correlations)}

    return d


def get_correlation_distances_dict(
    y_real: torch.Tensor, y_pred: torch.Tensor, stage: str, feature_names: Sequence
) -> Dict[str, int]:
    corr_real = corr(y_real)
    corr_pred = corr(y_pred)

    metric_names = [f"{stage}_corr_dist/{'-'.join(x)}" for x in combinations(feature_names, 2)]

    mse = nn.MSELoss(reduction="none")
    corr_distances = mse(corr_real, corr_pred).mean(dim=0)

    d = {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)}

    return d
