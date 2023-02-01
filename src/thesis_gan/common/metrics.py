from itertools import combinations
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import seaborn as sb
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from thesis_gan.common.utils import compute_avg_log_returns, compute_avg_volumes, corr, extract_data_stats

plt.rcParams["figure.figsize"] = [16, 9]
# plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.family"] = "serif"


# CORRELATIONS
def get_correlations_dict(y: torch.Tensor, real_o_pred: str, feature_names: Sequence) -> Dict[str, float]:
    # y.shape = [batch_size, n_features, decoder_length]
    correlations = corr(y).squeeze()
    metric_names = [f"{real_o_pred}_correlation/{'-'.join(x)}" for x in combinations(feature_names, 2)]
    d = {metric: correlation.item() for metric, correlation in zip(metric_names, correlations)}
    return d


# CORRELATION DISTANCES
def get_correlation_distances_dict(
    y_real: torch.Tensor, y_pred: torch.Tensor, stage: str, feature_names: Sequence
) -> Dict[str, float]:
    corr_real, corr_pred = corr(y_real), corr(y_pred)
    metric_names = [f"{stage}_corr_dist/{'-'.join(x)}" for x in combinations(feature_names, 2)]
    mse = nn.MSELoss(reduction="none")
    corr_distances = mse(corr_real, corr_pred).mean(dim=0)
    d = {metric: corr_dist.item() for metric, corr_dist in zip(metric_names, corr_distances)}
    return d


# VOLUMES METRICS
def get_metrics_listdict(ts_real: np.ndarray, ts_pred: np.ndarray, stock_names) -> Sequence[Dict[str, float]]:
    list_to_return = list()

    stat_names = ["Mean", "Std", "Kurtosis", "Skew"]
    stat_funcs = [np.mean, np.std, stats.kurtosis, stats.skew]

    for stat_name, stat_func in zip(stat_names, stat_funcs):
        d = dict()

        metrics_real = stat_func(ts_real, axis=1)
        metric_names = [f"Real Volume: {stat_name}/{stock_name}" for stock_name in stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

        metrics_pred = stat_func(ts_pred, axis=1)
        metric_names = [f"Pred Volume: {stat_name}/{stock_name}" for stock_name in stock_names]
        d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

        list_to_return.append(d)

    d = dict()

    metrics_real = ts_real.min(axis=1)
    metric_names = [f"Real Volume: Min/{stock_name}" for stock_name in stock_names]
    d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

    metrics_pred = ts_pred.min(axis=1)
    metric_names = [f"Pred Volume: Min/{stock_name}" for stock_name in stock_names]
    d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

    metrics_real = ts_real.max(axis=1)
    metric_names = [f"Real Volume: Max/{stock_name}" for stock_name in stock_names]
    d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_real)})

    metrics_pred = ts_pred.max(axis=1)
    metric_names = [f"Pred Volume: Max/{stock_name}" for stock_name in stock_names]
    d.update({metric_name: metric for metric_name, metric in zip(metric_names, metrics_pred)})

    list_to_return.append(d)

    return list_to_return


# PLOT TIMESERIES
def get_plot_timeseries_conv(
    real: np.ndarray,
    pred: np.ndarray,
    dataset_type: str,
    stock_names: Sequence[str],
    encoder_length: int,
    price_o_volume: str,
) -> Any:
    I, J = None, None
    if dataset_type == "multistock":
        I, J = 2, 2
    if dataset_type == "DowJones":
        I, J = 5, 6

    history_indexes = np.arange(encoder_length)
    continuation_indexes = np.arange(encoder_length, real.shape[-1])

    fig, axes = plt.subplots(I, J)
    legend_elements = [
        Line2D([0], [0], color="C0", lw=2, label="Observed"),
        Line2D([0], [0], color="C1", lw=2, label="Real continuation"),
        Line2D([0], [0], color="C2", lw=2, label="Predicted continuation"),
    ]

    for i in range(I):
        for j in range(J):
            linear_index = i * J + j
            axes[i, j].set_title(f"{stock_names[linear_index]}", fontsize=20)

            if dataset_type == "DowJones":
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
            else:
                axes[i, j].axvline(x=encoder_length, color="r")

            axes[i, j].plot(
                history_indexes,
                real[linear_index, :encoder_length],
                color="C0",
            )
            axes[i, j].plot(
                continuation_indexes,
                real[linear_index, encoder_length:],
                color="C1",
            )
            axes[i, j].plot(
                continuation_indexes,
                pred[linear_index, encoder_length:],
                color="C2",
            )

    fig.suptitle(price_o_volume, fontsize=24, y=0.94)
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1))
    fig.tight_layout()

    return fig


def get_plot_timeseries_timegan(
    real: np.ndarray, pred: np.ndarray, dataset_type: str, stock_names: Sequence[str], price_o_volume: str, stage: str
) -> Any:
    I, J = None, None
    if dataset_type == "multistock":
        I, J = 2, 2
    if dataset_type == "DowJones":
        I, J = 5, 6

    indexes = np.arange(real.shape[1])

    fig, axes = plt.subplots(I, J)
    legend_elements = [
        Line2D([0], [0], color="C1", lw=2, label="Real"),
        Line2D([0], [0], color="C2", lw=2, label="Predicted"),
    ]

    for i in range(I):
        for j in range(J):
            linear_index = i * J + j
            axes[i, j].set_title(f"{stock_names[linear_index]}", fontsize=20)

            if dataset_type == "DowJones":
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])

            axes[i, j].plot(
                indexes,
                real[linear_index],
                color="C1",
            )
            axes[i, j].plot(
                indexes,
                pred[linear_index],
                color="C2",
            )

    fig.suptitle(f"{price_o_volume} - {stage}", fontsize=24, y=0.94)
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1))
    fig.tight_layout()

    return fig


# PLOT RETURNS DISTRIBUTION
def draw_hist(ax, col_real, col_pred, stock_name, xlim=(-0.02, 0.02)):
    sb.histplot(data=col_real, kde=True, color="orange", legend=True, ax=ax)
    sb.histplot(data=col_pred, kde=True, color="green", legend=True, ax=ax)

    mu, sigma, rtn_range, norm_pdf = extract_data_stats(col_real)
    ax.plot(rtn_range, norm_pdf, "orange", lw=3, label=f"Real: N({mu:.5f}, {sigma**2:.5f})")

    mu, sigma, rtn_range, norm_pdf = extract_data_stats(col_pred)
    ax.plot(rtn_range, norm_pdf, "green", lw=3, label=f"Pred: N({mu:.5f}, {sigma**2:.5f})")

    ax.axvline(x=0, c="c", linestyle="--", lw=3)
    ax.set_title(f"{stock_name}", fontsize=24)
    ax.set_xlim(xlim)
    ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)


def compute_returns_distribution(ax, prices_real, prices_pred, stock_name):
    prices_real = pd.DataFrame(prices_real, columns=["mid_price"])
    prices_pred = pd.DataFrame(prices_pred, columns=["mid_price"])
    prices_real["Returns"] = prices_real["mid_price"].pct_change()
    prices_pred["Returns"] = prices_pred["mid_price"].pct_change()
    prices_real = prices_real.dropna()
    prices_pred = prices_pred.dropna()
    draw_hist(ax, prices_real["Returns"], prices_pred["Returns"], stock_name)


def get_plot_sf_returns_distribution(
    prices_real: np.ndarray,
    prices_pred: np.ndarray,
    stock_names,
) -> Any:

    fig, axes = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            linear_index = i * 2 + j
            compute_returns_distribution(
                axes[i, j], prices_real[linear_index, :], prices_pred[linear_index, :], stock_names[linear_index]
            )

    fig.suptitle("Returns distribution", fontsize=24)
    fig.tight_layout()

    return fig


# PLOT AGGREGATIONAL GAUSSIANITY
def draw_hist_multi(col_real, col_pred, xlim=(-0.02, 0.02), ax=None):
    sb.histplot(data=col_real, kde=True, color="orange", legend=True, ax=ax)
    sb.histplot(data=col_pred, kde=True, color="green", legend=True, ax=ax)

    mu, sigma, rtn_range, norm_pdf = extract_data_stats(col_real)
    ax.plot(rtn_range, norm_pdf, "orange", lw=3, label=f"Real: N({mu:.5f}, {sigma**2:.5f})")

    mu, sigma, rtn_range, norm_pdf = extract_data_stats(col_pred)
    ax.plot(rtn_range, norm_pdf, "green", lw=3, label=f"Pred: N({mu:.5f}, {sigma**2:.5f})")

    ax.set_xlim(xlim)
    ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)


def get_aggregational_gaussianities(prices_real, prices_pred):

    prices_real = pd.DataFrame(prices_real, columns=["mid_price"])
    prices_pred = pd.DataFrame(prices_pred, columns=["mid_price"])

    df_simple_rtn_real = pd.DataFrame(prices_real["mid_price"])
    df_simple_rtn_pred = pd.DataFrame(prices_pred["mid_price"])

    lags = 6
    cols = list()
    for lag in range(1, lags + 1):
        col = f"Returns - Lag {lag}"
        cols.append(col)
        df_simple_rtn_real[col] = df_simple_rtn_real["mid_price"].pct_change(periods=lag)
        df_simple_rtn_pred[col] = df_simple_rtn_pred["mid_price"].pct_change(periods=lag)

    df_simple_rtn_real.dropna(inplace=True)
    df_simple_rtn_pred.dropna(inplace=True)

    df_simple_rtn_real = df_simple_rtn_real.drop(["mid_price"], axis=1)
    df_simple_rtn_pred = df_simple_rtn_pred.drop(["mid_price"], axis=1)

    df_simple_rtn_real.columns = cols
    df_simple_rtn_pred.columns = cols

    return df_simple_rtn_real, df_simple_rtn_pred


def get_plot_sf_aggregational_gaussianity(prices: np.array, pred_prices: np.array, stock_name: str) -> Any:
    fig, axs = plt.subplots(nrows=2, ncols=3)
    df_real, df_pred = get_aggregational_gaussianities(prices, pred_prices)

    axs = axs.ravel()
    for i, col in enumerate(df_real.columns):
        draw_hist_multi(df_real[col], df_pred[col], ax=axs[i])

    fig.suptitle(f"Distribution of returns with increased time scale - {stock_name}", fontsize=24)
    fig.tight_layout()

    return fig


# PLOT ABSENCE AUTOCORRELATION
def corr_plot(corr, ax, title):
    sb.set(style="white")
    cmap = sb.diverging_palette(220, 20, as_cmap=True)
    sb.heatmap(corr, annot=True, cmap=cmap, square=True, linewidths=3, linecolor="w", ax=ax)
    ax.set_title(title, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment="center")


def get_plot_sf_absence_autocorrelation(prices_real: np.array, prices_pred: np.array, stock_name: str) -> Any:
    prices_real = pd.DataFrame(prices_real, columns=["mid_price"])
    prices_pred = pd.DataFrame(prices_pred, columns=["mid_price"])

    df_simple_rtn_real = pd.DataFrame(prices_real["mid_price"])
    df_simple_rtn_pred = pd.DataFrame(prices_pred["mid_price"])

    lags = 6
    cols = list()
    for lag in range(1, lags + 1):
        col = f"Lag {lag}"
        cols.append(col)
        df_simple_rtn_real[col] = df_simple_rtn_real["mid_price"].pct_change(periods=lag)
        df_simple_rtn_pred[col] = df_simple_rtn_pred["mid_price"].pct_change(periods=lag)

    df_simple_rtn_real.dropna(inplace=True)
    df_simple_rtn_pred.dropna(inplace=True)

    df_simple_rtn_real = df_simple_rtn_real.drop(["mid_price"], axis=1)
    df_simple_rtn_pred = df_simple_rtn_pred.drop(["mid_price"], axis=1)

    df_simple_rtn_real.columns = cols
    df_simple_rtn_pred.columns = cols

    corr_real = df_simple_rtn_real.corr()
    corr_pred = df_simple_rtn_pred.corr()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    axs = axs.ravel()
    corr_plot(corr_real, ax=axs[0], title="Real - Returns")
    corr_plot(corr_pred, ax=axs[1], title="Pred - Returns")

    fig.suptitle(f"Returns Autocorrelations - {stock_name}", fontsize=24)
    fig.tight_layout()

    return fig


# PLOT VOLATILITY CLUSTERING
def compute_volatility_clustering(ax, prices_real, prices_pred, stock_name):
    prices_real = pd.DataFrame(prices_real, columns=["mid_price"])
    prices_pred = pd.DataFrame(prices_pred, columns=["mid_price"])

    prices_real["Returns"] = prices_real["mid_price"].pct_change()
    prices_pred["Returns"] = prices_pred["mid_price"].pct_change()

    prices_real = prices_real.dropna()
    prices_pred = prices_pred.dropna()

    ax.plot(prices_real["Returns"], label="Real", color="C1")
    ax.plot(prices_pred["Returns"], label="Pred", color="C2", alpha=0.3)

    ax.set_ylabel("Returns")
    ax.set_title(stock_name, fontsize=20)


def get_plot_sf_volatility_clustering(prices, pred_prices, stock_names):
    fig, axs = plt.subplots(2, 2)
    legend_elements = [
        Line2D([0], [0], color="C1", lw=2, label="Real"),
        Line2D([0], [0], color="C2", alpha=0.3, lw=2, label="Synthetic"),
    ]

    for i in range(2):
        for j in range(2):
            linear_index = i * 2 + j
            compute_volatility_clustering(
                axs[i, j], prices[linear_index, :], pred_prices[linear_index, :], stock_names[linear_index]
            )

    fig.suptitle("Volatility clustering", fontsize=24, y=0.94)
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=15, bbox_to_anchor=(0.5, 1))
    fig.tight_layout()

    return fig


# PLOT VOLUME VOLATILITY CORRELATION
def get_plot_sf_volume_volatility_correlation(x_price, x_hat_price, x_volume, x_hat_volume, stock_names, delta):
    real_avg_log_returns = compute_avg_log_returns(x_price, delta)
    real_avg_volumes = compute_avg_volumes(x_volume, delta)
    pred_avg_log_returns = compute_avg_log_returns(x_hat_price, delta)
    pred_avg_volumes = compute_avg_volumes(x_hat_volume, delta)

    fig, ax = plt.subplots(2, 4)

    for target_idx in range(4):
        stock_name = stock_names[target_idx]

        # Real volume-volatility correlation
        title = f"{stock_name} - Real"
        ax[0, target_idx].set_title(title)
        ax[0, target_idx].scatter(
            real_avg_log_returns[target_idx],
            real_avg_volumes[target_idx],
            color="C0",
        )
        ax[0, target_idx].set_xlabel("Avg log-returns")
        ax[0, target_idx].set_ylabel("Avg log-volumes")

        # Pred volume-volatility correlation
        title = f"{stock_name} - Pred"
        ax[1, target_idx].set_title(title)
        ax[1, target_idx].scatter(
            pred_avg_log_returns[target_idx],
            pred_avg_volumes[target_idx],
            color="C1",
        )
        ax[1, target_idx].set_xlabel("Avg log-returns")
        ax[1, target_idx].set_ylabel("Avg log-volumes")

    fig.suptitle("Volume-Volatility Correlation", fontsize=24)
    fig.tight_layout()

    return fig
