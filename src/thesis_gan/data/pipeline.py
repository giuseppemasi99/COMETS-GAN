from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


def is_fitted(scaler: Union[MinMaxScaler, StandardScaler]) -> bool:
    try:
        check_is_fitted(scaler)
        return True
    except NotFittedError:
        return False


class Pipeline:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def preprocess(self, df: pd.DataFrame, targets: List[str]) -> np.ndarray:
        pass

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (scaler={self.scaler})"


class ScalerPipeline(Pipeline):
    def __init__(self, scaler: Union[MinMaxScaler, StandardScaler], round: bool, log: bool) -> None:
        super(ScalerPipeline, self).__init__()
        self.scaler = scaler
        self.round = round
        self.log = log

        self.first_volumes = None

    def preprocess(self, df: pd.DataFrame, targets: List[str]) -> np.ndarray:
        df_targets = df[targets].to_numpy()
        self.first_volumes = df_targets[0]

        df_targets = df_targets[1:]

        if self.log:
            df_targets = np.log1p(df_targets)

        if not is_fitted(self.scaler):
            self.scaler.fit(df_targets)

        return self.scaler.transform(df_targets)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        inv = self.scaler.inverse_transform(x)
        if self.log:
            inv = np.expm1(inv)
        if self.round:
            inv = np.rint(inv)
        inv = np.nan_to_num(inv)
        inv = np.vstack([self.first_volumes, inv])

        return inv


class ReturnPipeline(Pipeline):
    def __init__(self, scaler: Union[MinMaxScaler, StandardScaler]) -> None:
        super(ReturnPipeline, self).__init__()
        self.scaler = scaler
        self.first_prices = None

    def preprocess(self, df: pd.DataFrame, targets: List[str]) -> np.ndarray:
        prices = df[targets]
        self.first_prices = prices.iloc[0].values

        returns = prices.pct_change()[1:].to_numpy()

        if not is_fitted(self.scaler):
            self.scaler.fit(returns)

        return self.scaler.transform(returns)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        returns = self.scaler.inverse_transform(x)
        prices = np.cumprod(1 + returns, axis=0) * self.first_prices
        prices = np.nan_to_num(prices)
        prices = np.vstack([self.first_prices, prices])
        return prices


class LogReturnPipeline(Pipeline):
    def __init__(self, scaler: Union[MinMaxScaler, StandardScaler]) -> None:
        super(LogReturnPipeline, self).__init__()
        self.scaler = scaler
        self.first_prices = None

    def preprocess(self, df: pd.DataFrame, targets: List[str]) -> np.ndarray:
        prices = df[targets]
        self.first_prices = prices.iloc[0].values

        returns = prices.pct_change()[1:].to_numpy()
        log_returns = np.log1p(returns)

        if not is_fitted(self.scaler):
            self.scaler.fit(log_returns)

        return self.scaler.transform(log_returns)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        log_returns = self.scaler.inverse_transform(x)
        returns = np.expm1(log_returns)
        prices = np.cumprod(1 + returns, axis=0)
        prices = prices * self.first_prices
        # prices = np.nan_to_num(prices)
        prices[np.isinf(prices)] = 1e19
        prices = np.vstack([self.first_prices, prices])
        return prices
