import pandas as pd
import numpy as np
import re
import math
import datetime
from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from arch import arch_model
from collections import defaultdict
from scipy.stats import norm
from portfolio_management.utils import _filter_columns_and_indexes

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")


def calc_strategy_oos(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    expanding: bool = False,
    lag_number: int = 1,
    weight_multiplier: float = 100,
    weight_min: Union[None, float] = None,
    weight_max: Union[None, float] = None,
    name: str = None,
):
    """
    Calculates an out-of-sample strategy based on rolling or expanding window regression.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (strategy returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    expanding (bool, default=False): If True, uses an expanding window instead of rolling.
    lag_number (int, default=1): Number of lags to apply to the predictors.
    weight_multiplier (float, default=100): Multiplier to adjust strategy weights.
    weight_min (float or None, default=None): Minimum allowable weight.
    weight_max (float or None, default=None): Maximum allowable weight.
    name (str, default=None): Name for labeling the strategy returns.

    Returns:
    pd.DataFrame: Time series of strategy returns.
    """
    raise NotImplementedError("Function not available")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    replication_oos = calc_replication_oos(
        y=y,
        X=X,
        intercept=intercept,
        rolling_size=rolling_size,
        lag_number=lag_number,
        expanding=expanding,
    )
    actual_returns = replication_oos["Actual"]
    predicted_returns = replication_oos["Prediction"]
    strategy_weights = predicted_returns * weight_multiplier
    weight_min = weight_min if weight_min is not None else strategy_weights.min()
    weight_max = weight_max if weight_max is not None else strategy_weights.max()
    strategy_weights = strategy_weights.clip(lower=weight_min, upper=weight_max)
    strategy_returns = (actual_returns * strategy_weights).to_frame()
    if name:
        strategy_returns.columns = [name]
    else:
        strategy_returns.columns = [f"{y.columns[0]} Strategy"]
    return strategy_returns


def calc_replication_oos_not_lagged_features(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    return_r_squared_oos: float = False,
    r_squared_time_series: bool = False,
    return_parameters: bool = True,
    oos: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Performs out-of-sample replication without lagged features.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (actual returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    return_r_squared_oos (float, default=False): If True, returns the out-of-sample R-squared.
    r_squared_time_series (bool, default=False): If True, calculates time-series R-squared.
    return_parameters (bool, default=True): If True, returns regression parameters.
    oos (int, default=1): Number of periods for out-of-sample evaluation.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics for the out-of-sample replication.
    """
    raise NotImplementedError("Function not available")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    if "date" in X.columns.str.lower():
        X = X.rename({"Date": "date"}, axis=1)
        X = X.set_index("date")
    X.index.name = "date"

    oos_print = "In-Sample" if oos == 0 else f"{oos}OS"

    summary = defaultdict(list)

    if isinstance(y, pd.Series):
        y = pd.DataFrame(y)
    y_name = y.columns[0]

    for idx in range(rolling_size, len(y.index) + 1 - oos, 1):
        X_rolling = X.iloc[idx - rolling_size : idx].copy()
        y_rolling = y.iloc[idx - rolling_size : idx, 0].copy()

        y_oos = y.iloc[idx - 1 + oos, 0].copy()
        X_oos = X.iloc[idx - 1 + oos, :].copy()

        if intercept:
            X_rolling = sm.add_constant(X_rolling)

        try:
            regr = sm.OLS(
                y_rolling, X_rolling, missing="drop", hasconst=intercept
            ).fit()
        except ValueError:
            y_rolling = y_rolling.reset_index(drop=True)
            X_rolling = X_rolling.reset_index(drop=True)
            regr = sm.OLS(
                y_rolling, X_rolling, missing="drop", hasconst=intercept
            ).fit()

        for jdx, coeff in enumerate(regr.params.index):
            if coeff != "const":
                summary[f"{coeff} Beta {oos_print}"].append(regr.params[jdx])
            else:
                summary[f"{coeff} {oos_print}"].append(regr.params[jdx])

        if intercept:
            y_pred = regr.params[0] + (regr.params[1:] @ X_oos)
        else:
            y_pred = regr.params @ X_oos

        summary[f"{y_name} Replicated"].append(y_pred)
        summary[f"{y_name} Actual"].append(y_oos)

    summary = pd.DataFrame(summary, index=X.index[rolling_size - 1 + oos :])

    if r_squared_time_series:
        time_series_error = pd.DataFrame({})
        for idx in range(rolling_size, len(y.index) + 1 - oos, 1):
            y_rolling = y.iloc[idx - rolling_size : idx, 0].copy()
            y_oos = y.iloc[idx - 1 + oos, 0].copy()
            time_series_error.loc[y.index[idx - 1 + oos], "Naive Error"] = (
                y_oos - y_rolling.mean()
            )
        time_series_error["Model Error"] = (
            summary[f"{y_name} Actual"] - summary[f"{y_name} Replicated"]
        )
        oos_rsquared = (
            1
            - time_series_error["Model Error"].apply(lambda x: x**2).sum()
            / time_series_error["Naive Error"].apply(lambda x: x**2).sum()
        )
    else:
        oos_rsquared = (
            1
            - (summary[f"{y_name} Actual"] - summary[f"{y_name} Replicated"]).var()
            / summary[f"{y_name} Actual"].var()
        )

    if return_r_squared_oos:
        return oos_rsquared

    if not return_parameters:
        summary = summary[[f"{y_name} Actual", f"{y_name} Replicated"]]

    if not intercept:
        summary = summary.rename(
            columns=lambda c: c.replace(" Replicated", f" Replicated no Intercept")
        )

    if not intercept:
        print(f"R^Squared {oos_print} without Intercept: {oos_rsquared:.2%}")

    else:
        print(f"R^Squared {oos_print}: {oos_rsquared:.2%}")

    summary = summary.rename(
        columns=lambda c: (
            c.replace(" Replicated", f" Replicated {oos_print}").replace(
                " Actual", f" Actual {oos_print}"
            )
        )
    )

    return _filter_columns_and_indexes(
        summary,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_replication_oos(
    y: Union[pd.Series, pd.DataFrame],
    X: Union[pd.Series, pd.DataFrame],
    intercept: bool = True,
    rolling_size: Union[None, int] = 60,
    expanding: bool = False,
    return_r_squared_oos: float = False,
    lag_number: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Performs out-of-sample replication of a time series regression with rolling or expanding windows.

    Parameters:
    y (pd.Series or pd.DataFrame): Dependent variable (actual returns).
    X (pd.Series or pd.DataFrame): Independent variable(s) (predictors).
    intercept (bool, default=True): If True, includes an intercept in the regression.
    rolling_size (int or None, default=60): Size of the rolling window for in-sample fitting.
    expanding (bool, default=False): If True, uses an expanding window instead of rolling.
    return_r_squared_oos (float, default=False): If True, returns the out-of-sample R-squared.
    lag_number (int, default=1): Number of lags to apply to the predictors.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics for the out-of-sample replication.
    """
    raise NotImplementedError("Function not available")
    try:
        y = y.copy()
        X = X.copy()
    except:
        pass
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    if "date" in X.columns.str.lower():
        X = X.rename({"Date": "date"}, axis=1)
        X = X.set_index("date")
    X.index.name = "date"

    X = X.shift(lag_number)

    df = y.join(X, how="inner")
    y = df.iloc[:, [0]].copy()
    X = df.iloc[:, 1:].copy()

    if intercept:
        X = sm.add_constant(X)

    summary_pred = pd.DataFrame({})

    for i, last_is_date in enumerate(y.index):
        if i < (rolling_size):
            continue
        y_full = y.iloc[:i].copy()
        if expanding:
            y_rolling = y_full.copy()
        else:
            y_rolling = y_full.iloc[-rolling_size:]
        X_full = X.iloc[:i].copy()
        if expanding:
            X_rolling = X_full.copy()
        else:
            X_rolling = X_full.iloc[-rolling_size:]

        reg = sm.OLS(y_rolling, X_rolling, hasconst=intercept, missing="drop").fit()
        y_pred = reg.predict(X.iloc[i, :])
        naive_y_pred = y_full.mean()
        y_actual = y.iloc[i]
        summary_line = (
            reg.params.to_frame()
            .transpose()
            .rename(
                columns=lambda c: (
                    c.replace("const", "Alpha") if c == "const" else c + " Lag Beta"
                )
            )
        )
        summary_line["Prediction"] = y_pred[0]
        summary_line["Naive Prediction"] = naive_y_pred.squeeze()
        summary_line["Actual"] = y_actual.squeeze()
        summary_line.index = [y.index[i]]
        summary_pred = pd.concat([summary_pred, summary_line], axis=0)

    summary_pred["Prediction Error"] = (
        summary_pred["Prediction"] - summary_pred["Actual"]
    )
    summary_pred["Naive Prediction Error"] = (
        summary_pred["Naive Prediction"] - summary_pred["Actual"]
    )

    rss = (np.array(summary_pred["Prediction Error"]) ** 2).sum()
    tss = (np.array(summary_pred["Naive Prediction Error"]) ** 2).sum()

    oos_rsquared = 1 - rss / tss

    if return_r_squared_oos:
        return pd.DataFrame(
            {"R^Squared OOS": oos_rsquared},
            index=[
                y.columns[0]
                + " ~ "
                + " + ".join(
                    [
                        c.replace("const", "Alpha") if c == "const" else c + " Lag Beta"
                        for c in X.columns
                    ]
                )
            ],
        )

    print("OOS R^Squared: {:.4%}".format(oos_rsquared))

    return _filter_columns_and_indexes(
        summary_pred,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_rolling_oos_port(
    returns: pd.DataFrame,
    weights_func,
    window: Union[None, int] = None,
    weights_func_params: dict = {},
    port_name: str = "Portfolio OOS",
    expanding: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates a rolling out-of-sample portfolio based on a rolling or expanding window optimization.

    Parameters:
    returns (pd.DataFrame): Time series of asset returns.
    weights_func (function): Function to calculate the portfolio weights.
    window (int or None, default=None): Rolling window size for in-sample optimization.
    weights_func_params (dict, default={}): Additional parameters for the weights function.
    port_name (str, default='Portfolio OOS'): Name for the portfolio.
    expanding (bool, default=False): If True, uses an expanding window instead of a rolling one.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Out-of-sample portfolio returns.
    """
    raise NotImplementedError("Function not available")
    if window is None:
        print(
            'Using "window" of 60 periods for in-sample optimization, since none were provided.'
        )
        window = 60
    returns = returns.copy()
    if "date" in returns.columns.str.lower():
        returns = returns.rename({"Date": "date"}, axis=1)
        returns = returns.set_index("date")
    returns.index.name = "date"

    port_returns_oos = pd.DataFrame({})

    for idx in range(0, len(returns.index) - window):
        modified_idx = 0 if expanding else idx
        weights_func_all_params = {
            "returns": returns.iloc[modified_idx : (window + idx), :]
        }
        weights_func_all_params.update(weights_func_params)
        wts = weights_func(**weights_func_all_params).iloc[:, 0]
        idx_port_return_oos = sum(returns.iloc[window, :].loc[wts.index] * wts)
        idx_port_return_oos = pd.DataFrame(
            {port_name: idx_port_return_oos}, index=[returns.index[idx + window]]
        )
        port_returns_oos = pd.concat([port_returns_oos, idx_port_return_oos])

    return _filter_columns_and_indexes(
        port_returns_oos,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )
