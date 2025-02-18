import pandas as pd
import numpy as np
import re
import math
import datetime
from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from warnings import filterwarnings, warn
from arch import arch_model
from collections import defaultdict
from scipy.stats import norm
from portfolio_management.utils import _filter_columns_and_indexes, clean_returns_df

pd.options.display.float_format = "{:,.4f}".format

DEFAULT_WINDOW_VAR_CALCULATION = 60
DEFAULT_EWMA_THETA = 0.94
DELTA_EWMA_INITIAL_VOL = 0.2 / np.sqrt(252)


def calc_ewma_volatility(
    excess_returns: pd.Series,
    theta: float = DEFAULT_EWMA_THETA,
    initial_vol: float = DELTA_EWMA_INITIAL_VOL,
) -> pd.Series:
    """
    Calculates the Exponentially Weighted Moving Average (EWMA) volatility of excess returns.

    Parameters
    ----------
    excess_returns : pd.Series
        Time series of excess returns.
    theta : float, optional
        Decay factor for the EWMA. A value closer to 1 discounts older observations more slowly.
        Defaults to 0.94.
    initial_vol : float, optional
        Initial volatility (annualized) used to start the EWMA calculation. Defaults to 0.2 / sqrt(252).

    Returns
    -------
    pd.Series
        A series of EWMA volatilities, indexed by the same dates as `excess_returns`.

    Notes
    -----
    - This function uses a recursive formula:
    var(t) = theta * var(t-1) + (1 - theta) * (excess_returns(t))^2
    - The square root of var(t) is taken to get the volatility at time t.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.risk import calc_ewma_volatility
    >>> returns = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2023-01-01", periods=3))
    >>> ewma_vol = calc_ewma_volatility(returns, theta=0.94)
    >>> ewma_vol
    2023-01-01    0.010000
    2023-01-02    0.015524
    2023-01-03    0.014358
    dtype: float64
    """
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
    Calculates GARCH-based conditional volatility for a given series of excess returns.

    Parameters
    ----------
    excess_returns : pd.Series
        Time series of excess returns.
    p : int, optional
        The order of the GARCH component (lagged variance terms). Defaults to 1.
    q : int, optional
        The order of the ARCH component (lagged squared returns). Defaults to 1.

    Returns
    -------
    pd.Series
        A series representing the GARCH model's conditional volatility, indexed by the same dates
        as `excess_returns`.

    Raises
    ------
    ValueError
        If `excess_returns` is empty.
    RuntimeError
        If the GARCH model fails to fit properly.

    Notes
    -----
    - The function rescales the input series if its standard deviation is very small,
    to improve numerical stability when fitting the GARCH model.
    - The fitted volatility is then rescaled back to the original scale.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.risk import calc_garch_volatility
    >>> returns = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2023-01-01", periods=3))
    >>> garch_vol = calc_garch_volatility(returns, p=1, q=1)
    >>> garch_vol
    2023-01-01    0.010000
    2023-01-02    0.014652
    2023-01-03    0.012897
    dtype: float64
    """
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
    warnings: bool = True,
):
    """
    Calculates a summary of VaR (Value at Risk) and CVaR (Conditional VaR) for the provided returns,
    with optional volatility estimates and hit ratio calculations.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Time series of returns. If a DataFrame is provided, only the first column is used.
    quantile : float or None, optional
        Quantile for VaR/CVaR calculations (e.g., 0.05 for 5% VaR). Must be between 0 and 1.
        Defaults to 0.05.
    window : int, optional
        Rolling window size for historical calculations. Defaults to 60.
    return_hit_ratio : bool, optional
        If True, returns a DataFrame containing the hit ratio (frequency of VaR breaches).
        Defaults to False.
    filter_first_hit_ratio_date : str, datetime.date, or None, optional
        Earliest date to include for hit ratio calculations. Defaults to None.
    return_stats : str or list, optional
        Which statistics to return. Options include 'Returns', 'VaR', 'CVaR', 'Vol'.
        Defaults to ['Returns', 'VaR', 'CVaR', 'Vol'].
    full_time_sample : bool, optional
        If True, only returns the expanding (cumulative) metrics. Defaults to False.
    z_score : float, optional
        Custom z-score for parametric VaR. If None, uses `norm.ppf(quantile)`. Defaults to None.
    shift : int, optional
        Number of periods to shift the VaR/CVaR calculations (e.g., 1 for a 1-day-ahead VaR). Defaults to 1.
    normal_vol_formula : bool, optional
        If True, uses standard deviation for volatility. If False, uses sqrt of mean squared returns.
        Defaults to False.
    ewma_theta : float, optional
        Decay factor for EWMA volatility calculations. Defaults to 0.94.
    ewma_initial_vol : float, optional
        Initial volatility for EWMA. Defaults to 0.2 / sqrt(252).
    garch_p : int, optional
        Order of the GARCH model's lagged variance terms. Defaults to 1.
    garch_q : int, optional
        Order of the GARCH model's lagged squared returns. Defaults to 1.
    keep_columns : list or str, optional
        Columns to keep in the final DataFrame. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the final DataFrame. Defaults to None.
    keep_indexes : list or str, optional
        Indexes (rows) to keep in the final DataFrame. Defaults to None.
    drop_indexes : list or str, optional
        Indexes (rows) to drop from the final DataFrame. Defaults to None.
    drop_before_keep : bool, optional
        If True, drops specified columns/indexes before keeping. Defaults to False.
    warnings : bool, optional
        If True, prints warnings when data is insufficient for reliable calculations. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - Returns (optional)
        - Historical and Parametric VaR
        - Historical and Parametric CVaR
        - Volatility estimates (expanding, rolling, EWMA, GARCH)
        - (Optional) a DataFrame of hit ratios, if `return_hit_ratio` is True.

    Raises
    ------
    ValueError
        If `quantile` is not between 0 and 1.
    ValueError
        If `shift` is negative.
    ValueError
        If `window` is less than 1.

    Notes
    -----
    - Historical VaR/CVaR is computed using rolling or expanding quantiles of returns.
    - Parametric VaR/CVaR is computed using volatility estimates (expanding, rolling, EWMA, GARCH)
    multiplied by the z-score.
    - Hit Ratio measures how often the actual loss exceeds the VaR forecast.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.risk import calc_var_cvar_summary
    >>> returns = pd.Series([0.01, -0.02, 0.005, 0.02],
    ...                     index=pd.date_range("2023-01-01", periods=4))
    >>> summary_df = calc_var_cvar_summary(returns, quantile=0.05, window=3)
    >>> summary_df.head()
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
            if warnings:
                warn(
                    "There are few data points to calculate the hit ratio, which might produce unreliable results."
                    + "Set 'warnings = False' to silence this message"
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
            if warnings:
                warn(
                    f"VaR and CVaR are given shifted by {shift:.0f} period."
                    + "Set 'warnings = False' to silence this message"
                )

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
