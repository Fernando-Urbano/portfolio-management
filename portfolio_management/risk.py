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
from portfolio_management.utils import _filter_columns_and_indexes, clean_returns_df

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")

DEFAULT_WINDOW_VAR_CALCULATION = 60
DEFAULT_EWMA_THETA = 0.94
DELTA_EWMA_INITIAL_VOL = 0.2 / np.sqrt(252)


def calc_ewma_volatility(
    excess_returns: pd.Series,
    theta: float = DEFAULT_EWMA_THETA,
    initial_vol: float = DELTA_EWMA_INITIAL_VOL,
) -> pd.Series:
    var_t0 = initial_vol**2
    ewma_var = [var_t0]
    for i in range(len(excess_returns.index)):
        new_ewma_var = ewma_var[-1] * theta + (excess_returns.iloc[i] ** 2) * (
            1 - theta
        )
        ewma_var.append(new_ewma_var)
    ewma_var.pop(0)  # Remove var_t0
    ewma_vol = [np.sqrt(v) for v in ewma_var]
    return pd.Series(ewma_vol, index=excess_returns.index)


def calc_garch_volatility(
    excess_returns: pd.Series, p: int = 1, q: int = 1
) -> pd.Series:
    """
    Calculate GARCH volatility for a given series of excess returns with automatic scaling.

    Parameters:
    ----------
    excess_returns : pd.Series
        Time series of excess returns.
    p : int, default=1
        Order of the GARCH model for the lagged variance terms.
    q : int, default=1
        Order of the GARCH model for the lagged squared returns.

    Returns:
    -------
    pd.Series
        Conditional volatility series corresponding to the input excess returns.

    Notes:
    -----
    - Automatically scales the input data to improve numerical stability.
    - Rescales the conditional volatility back to the original scale.
    """
    # Check if the series is empty
    if excess_returns.empty:
        raise ValueError("Input excess_returns series is empty.")
    std_dev = excess_returns.std()

    scaling_threshold = 1e-3

    scaling_factor = 1.0

    if std_dev < scaling_threshold and std_dev > 0:
        scaling_factor = 1 / std_dev
        scaled_returns = excess_returns * scaling_factor
    else:
        scaled_returns = excess_returns.copy()

    model = arch_model(scaled_returns, vol="Garch", p=p, q=q, rescale=False)

    try:
        fitted_model = model.fit(disp="off")
    except Exception as e:
        raise RuntimeError(f"GARCH model fitting failed: {e}")

    scaled_volatility = fitted_model.conditional_volatility

    volatility = scaled_volatility / scaling_factor

    volatility = pd.Series(volatility, index=excess_returns.index)

    return volatility


def calc_var_cvar_summary(
    returns: Union[pd.Series, pd.DataFrame],
    quantile: Union[None, float] = 0.05,
    window: Union[int] = DEFAULT_WINDOW_VAR_CALCULATION,
    return_hit_ratio: bool = False,
    filter_first_hit_ratio_date: Union[None, str, datetime.date] = None,
    return_stats: Union[str, list] = ["Returns", "VaR", "CVaR", "Vol"],
    full_time_sample: bool = False,
    z_score: float = None,
    shift: int = 1,
    normal_vol_formula: bool = False,
    ewma_theta: float = DEFAULT_EWMA_THETA,
    ewma_initial_vol: float = DELTA_EWMA_INITIAL_VOL,
    garch_p: int = 1,
    garch_q: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates a summary of VaR (Value at Risk) and CVaR (Conditional VaR) for the provided returns.

    Parameters:
    returns (pd.Series or pd.DataFrame): Time series of returns.
    quantile (float or None, default=0.05): Quantile to calculate the VaR and CVaR.
    window (str or None, default=None): Window size for rolling calculations.
    return_hit_ratio (bool, default=False): If True, returns the hit ratio for the VaR.
    return_stats (str or list, default=['Returns', 'VaR', 'CVaR', 'Vol']): Statistics to return in the summary.
    full_time_sample (bool, default=False): If True, calculates using the full time sample.
    z_score (float, default=None): Z-score for parametric VaR calculation.
    shift (int, default=1): Period shift for VaR/CVaR calculations.
    normal_vol_formula (bool, default=False): If True, uses the normal volatility formula.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary of VaR and CVaR statistics.
    """
    if quantile > 1 or quantile < 0:
        raise ValueError(
            "Quantile must be between 0 and 1, meaning that it should not be passed as percentage"
        )
    if shift < 0:
        raise ValueError("Shift must be greater than or equal to 0.")
    if window < 1:
        raise ValueError("Window must be greater than 0.")
    returns = clean_returns_df(returns)

    if isinstance(returns, pd.DataFrame):
        returns_series = returns.iloc[:, 0]
        returns_series.index = returns.index
        returns = returns_series.copy()

    summary = pd.DataFrame({})

    # Returns
    summary[f"Returns"] = returns

    # VaR
    summary[f"Expanding {window:.0f} Historical VaR ({quantile:.2%})"] = (
        returns.expanding(min_periods=window).quantile(quantile)
    )
    summary[f"Rolling {window:.0f} Historical VaR ({quantile:.2%})"] = returns.rolling(
        window=window
    ).quantile(quantile)
    if normal_vol_formula:
        summary[f"Expanding {window:.0f} Volatility"] = returns.expanding(window).std()
        summary[f"Rolling {window:.0f} Volatility"] = returns.rolling(window).std()
    else:
        summary[f"Expanding {window:.0f} Volatility"] = np.sqrt(
            (returns**2).expanding(window).mean()
        )
        summary[f"Rolling {window:.0f} Volatility"] = np.sqrt(
            (returns**2).rolling(window).mean()
        )
    summary[f"EWMA {ewma_theta:.2f} Volatility"] = calc_ewma_volatility(
        returns, theta=ewma_theta, initial_vol=ewma_initial_vol
    )
    summary[f"GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility"] = calc_garch_volatility(
        returns, p=garch_p, q=garch_q
    )

    z_score = norm.ppf(quantile) if z_score is None else z_score
    summary[f"Expanding {window:.0f} Parametric VaR ({quantile:.2%})"] = (
        summary[f"Expanding {window:.0f} Volatility"] * z_score
    )
    summary[f"Rolling {window:.0f} Parametric VaR ({quantile:.2%})"] = (
        summary[f"Rolling {window:.0f} Volatility"] * z_score
    )
    summary[f"EWMA {ewma_theta:.2f} Parametric VaR ({quantile:.2%})"] = (
        summary[f"EWMA {ewma_theta:.2f} Volatility"] * z_score
    )
    summary[f"GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({quantile:.2%})"] = (
        summary[f"GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility"] * z_score
    )

    if return_hit_ratio:
        shift_stats = [
            f"Expanding {window:.0f} Historical VaR ({quantile:.2%})",
            f"Rolling {window:.0f} Historical VaR ({quantile:.2%})",
            f"Expanding {window:.0f} Parametric VaR ({quantile:.2%})",
            f"Rolling {window:.0f} Parametric VaR ({quantile:.2%})",
            f"EWMA {ewma_theta:.2f} Parametric VaR ({quantile:.2%})",
            f"GARCH({garch_p:.0f}, {garch_q:.0f}) Parametric VaR ({quantile:.2%})",
        ]
        summary_shift = summary[["Returns"] + shift_stats].copy()
        if shift > 0:
            summary_shift[shift_stats] = summary_shift[shift_stats].shift(shift)
        if filter_first_hit_ratio_date:
            if isinstance(
                filter_first_hit_ratio_date, (datetime.date, datetime.datetime)
            ):
                filter_first_hit_ratio_date = filter_first_hit_ratio_date.strftime(
                    "%Y-%m-%d"
                )
            summary_shift = summary_shift.loc[filter_first_hit_ratio_date:]
        if len(summary_shift.index) < 20:
            warnings.warn(
                "There are few data points to calculate the hit ratio, which might produce unreliable results"
            )

        summary_shift = summary_shift.dropna(axis=0)
        summary_shift[shift_stats] = summary_shift[shift_stats].apply(
            lambda x: (x - summary_shift["Returns"]) > 0
        )
        hit_ratio = pd.DataFrame(
            summary_shift[shift_stats].mean(), columns=["Hit Ratio"]
        )
        hit_ratio["Hit Ratio Error"] = (hit_ratio["Hit Ratio"] - quantile) / quantile
        hit_ratio["Hit Ratio Absolute Error"] = abs(hit_ratio["Hit Ratio Error"])
        hit_ratio = hit_ratio.sort_values("Hit Ratio Absolute Error")
        return _filter_columns_and_indexes(
            hit_ratio,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )

    # CVaR
    summary[f"Expanding {window:.0f} Historical CVaR ({quantile:.2%})"] = (
        returns.expanding(window).apply(lambda x: x[x < x.quantile(quantile)].mean())
    )
    summary[f"Rolling {window:.0f} Historical CVaR ({quantile:.2%})"] = returns.rolling(
        window
    ).apply(lambda x: x[x < x.quantile(quantile)].mean())
    summary[f"Expanding {window:.0f} Parametrical CVaR ({quantile:.2%})"] = (
        -norm.pdf(z_score) / quantile * summary[f"Expanding {window:.0f} Volatility"]
    )
    summary[f"Rolling {window:.0f} Parametrical CVaR ({quantile:.2%})"] = (
        -norm.pdf(z_score) / quantile * summary[f"Rolling {window:.0f} Volatility"]
    )
    summary[f"EWMA {ewma_theta:.2f} Parametrical CVaR ({quantile:.2%})"] = (
        -norm.pdf(z_score) / quantile * summary[f"EWMA {ewma_theta:.2f} Volatility"]
    )
    summary[
        f"GARCH({garch_p:.0f}, {garch_q:.0f}) Parametrical CVaR ({quantile:.2%})"
    ] = (
        -norm.pdf(z_score)
        / quantile
        * summary[f"GARCH({garch_p:.0f}, {garch_q:.0f}) Volatility"]
    )

    if shift > 0:
        shift_columns = [
            c for c in summary.columns if not bool(re.search("returns", c))
        ]
        summary[shift_columns] = summary[shift_columns].shift(shift)
        if shift == 1:
            warnings.warn(f"VaR and CVaR are given shifted by {shift:.0f} period.")

    if full_time_sample:
        summary = summary.loc[
            :,
            lambda df: [
                c for c in df.columns if bool(re.search("expanding", c.lower()))
            ],
        ]
    return_stats = (
        [return_stats.lower()]
        if isinstance(return_stats, str)
        else [s.lower() for s in return_stats]
    )
    return_stats = list(map(lambda x: "volatility" if x == "vol" else x, return_stats))
    if return_stats == ["all"] or set(return_stats) == set(
        ["returns", "var", "cvar", "volatility"]
    ):
        return _filter_columns_and_indexes(
            summary,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )
    return _filter_columns_and_indexes(
        summary.loc[
            :,
            lambda df: df.columns.map(
                lambda c: bool(
                    re.search(r"\b" + r"\b|\b".join(return_stats) + r"\b", c.lower())
                )
            ),
        ],
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )
