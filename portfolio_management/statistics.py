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
from portfolio_management.utils import _filter_columns_and_indexes
from scipy.stats import norm
from portfolio_management.port_construction import calc_tangency_weights

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")


ANNUAL_FACTOR_MAP = {
    "D": 360,
    "DU": 252,
    "W": 52,
    "BM": 12,
    "ME": 12,
    "BQ": 4,
    "BA": 2,
    "A": 1,
}

import numpy as np
import pandas as pd

ANNUAL_FACTOR_MAP = {
    "D": 252,  # Daily
    "W": 52,  # Weekly
    "BM": 12,  # Monthly (Business Monthly)
    "ME": 12,  # Monthly (Month-End)
    "BQ": 4,  # Quarterly
    "BA": 2,  # Semiannual (Biannual or Annual if larger gaps)
    "A": 1,  # Annual
}


def _transform_annual_factor(freq) -> int:
    return ANNUAL_FACTOR_MAP.get(freq)


def _calc_annual_factor(dates) -> str:
    """
    Given a list/array-like of dates, attempt to infer the frequency
    (daily, weekly, monthly, etc.) and return one of the keys in
    ANNUAL_FACTOR_MAP:
        "D"  -> 252   (Daily)
        "W"  -> 52    (Weekly)
        "BM" -> 12    (Monthly, not necessarily month-end)
        "ME" -> 12    (Monthly, all month-end)
        "BQ" -> 4     (Quarterly)
        "BA" -> 2     (Semiannual, or even annual in practice)

    If fewer than 20 dates are provided, the function raises a ValueError
    forcing the user to specify the frequency manually.

    :param dates: A list/array-like of datetime objects (or date strings).
    :return: A string key from ANNUAL_FACTOR_MAP indicating the inferred frequency.
    """
    # Require at least 20 dates to auto-detect
    if len(dates) < 20:
        raise ValueError(
            "Not enough data points to auto-detect frequency. "
            "At least 20 dates are required, otherwise please specify the frequency."
        )

    dates = pd.to_datetime(dates)
    dates = np.sort(dates)

    day_diffs = np.diff(dates) / np.timedelta64(1, "D")  # length n-1
    median_gap = np.median(day_diffs)

    # ----- Frequency detection thresholds -----
    #
    # Typical day-gap heuristics (approximate):
    #
    #   < 2 days   => "D"  (Daily)
    #   < 10 days  => "W"  (Weekly)
    #   < 40 days  => "ME" or "BM" (Monthly)
    #   < 80 days  => "BQ" (Quarterly)
    #   < 200 days => "BA" (Semiannual)
    #   >= 200 days=> "A" (also used for ~annual in this map)
    #
    if median_gap < 2:
        max_gap = np.max(day_diffs)
        frequency = "DU" if max_gap > 2 else "D"
    elif median_gap < 10:
        frequency = "W"
    elif median_gap < 40:
        # Distinguish "month-end" vs. "business-monthly"
        is_month_end = all(d == (d + pd.tseries.offsets.MonthEnd(0)) for d in dates)
        frequency = "ME" if is_month_end else "BM"
    elif median_gap < 80:
        frequency = "BQ"
    elif median_gap < 200:
        frequency = "BA"
    else:
        frequency = "A"
    return _transform_annual_factor(frequency)


def calc_negative_pct(
    returns: Union[pd.DataFrame, pd.Series, list],
    calc_positive: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates the percentage of negative or positive returns in the provided data.

    Parameters:
    returns (pd.DataFrame, pd.Series, or list): Time series of returns.
    calc_positive (bool, default=False): If True, calculates the percentage of positive returns.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: A DataFrame with the percentage of negative or positive returns, number of returns, and the count of negative/positive returns.
    """
    returns = returns.copy()
    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(
                series, right_index=True, left_index=True, how="outer"
            )

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if "date" in returns.columns.str.lower():
        returns = returns.rename({"Date": "date"}, axis=1)
        returns = returns.set_index("date")

    returns.index.name = "date"

    returns = returns.apply(lambda x: x.astype(float))
    prev_len_index = returns.apply(lambda x: len(x))
    returns = returns.dropna(axis=0)
    new_len_index = returns.apply(lambda x: len(x))
    if not (prev_len_index == new_len_index).all():
        print("Some columns had NaN values and were dropped")
    if calc_positive:
        returns = returns.map(lambda x: 1 if x > 0 else 0)
    else:
        returns = returns.map(lambda x: 1 if x < 0 else 0)

    negative_statistics = returns.agg(["mean", "count", "sum"]).set_axis(
        ["% Negative Returns", "Nº Returns", "Nº Negative Returns"], axis=0
    )

    if calc_positive:
        negative_statistics = negative_statistics.rename(
            lambda i: i.replace("Negative", "Positive"), axis=0
        )

    return _filter_columns_and_indexes(
        negative_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_cumulative_returns(
    returns: Union[pd.DataFrame, pd.Series],
    return_plot: bool = True,
    fig_size: tuple = (7, 5),
    return_series: bool = False,
    name: str = None,
    timeframes: Union[None, dict] = None,
):
    """
    Calculates cumulative returns from a time series of returns.

    Parameters:
    returns (pd.DataFrame or pd.Series): Time series of returns.
    return_plot (bool, default=True): If True, plots the cumulative returns.
    fig_size (tuple, default=(7, 5)): Size of the plot for cumulative returns.
    return_series (bool, default=False): If True, returns the cumulative returns as a DataFrame.
    name (str, default=None): Name for the title of the plot or the cumulative return series.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate cumulative returns for each period.

    Returns:
    pd.DataFrame or None: Returns cumulative returns DataFrame if `return_series` is True.
    """
    if timeframes is not None:
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0] : timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0] :]
            elif timeframe[1]:
                timeframe_returns = returns.loc[: timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f"No returns for {name} timeframe")
            calc_cumulative_returns(
                timeframe_returns,
                return_plot=True,
                fig_size=fig_size,
                return_series=False,
                name=name,
                timeframes=None,
            )
        return
    returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    returns = returns.apply(lambda x: x.astype(float))
    returns = returns.apply(lambda x: x + 1)
    returns = returns.cumprod()
    returns = returns.apply(lambda x: x - 1)
    title = f"Cumulative Returns {name}" if name else "Cumulative Returns"
    if return_plot:
        # Add a first row with a value of zero only for plotting
        plot_returns = returns.copy()
        first_row_index = plot_returns.index[0] - (
            plot_returns.index[1] - plot_returns.index[0]
        )
        first_row = pd.DataFrame(
            [[0] * plot_returns.shape[1]],
            columns=plot_returns.columns,
            index=[first_row_index],
        )
        plot_returns = pd.concat([first_row, plot_returns])
        # Plot
        plot_returns.plot(
            title=title,
            figsize=fig_size,
            grid=True,
            xlabel="Date",
            ylabel="Cumulative Returns",
        )
    if return_series:
        return returns


def get_best_and_worst(
    summary_statistics: pd.DataFrame,
    stat: str = "Annualized Sharpe",
):
    """
    Identifies the best and worst assets based on a specified statistic.

    Parameters:
    summary_statistics (pd.DataFrame): DataFrame containing summary statistics.
    stat (str, default='Annualized Sharpe'): The statistic to compare assets by.
    return_df (bool, default=True): If True, returns a DataFrame with the best and worst assets.

    Returns:
    pd.DataFrame or None: DataFrame with the best and worst assets if `return_df` is True.
    """
    summary_statistics = summary_statistics.copy()

    if len(summary_statistics.index) < 2:
        raise Exception(
            '"summary_statistics" must have at least two lines in order to do comparison'
        )

    if stat not in summary_statistics.columns:
        raise ValueError(f'{stat} not in "summary_statistics"')
    summary_statistics.rename(columns=lambda c: c.replace(" ", "").lower())
    best_stat = summary_statistics[stat].max()
    worst_stat = summary_statistics[stat].min()
    if all(pd.isna(summary_statistics[stat])):
        raise ValueError(f'All values in "{stat}" are missing')
    asset_best_stat = summary_statistics.loc[
        lambda df: df[stat] == df[stat].max()
    ].index[0]
    asset_worst_stat = summary_statistics.loc[
        lambda df: df[stat] == df[stat].min()
    ].index[0]
    return pd.concat(
        [
            summary_statistics.loc[lambda df: df.index == asset_best_stat],
            summary_statistics.loc[lambda df: df.index == asset_worst_stat],
        ]
    )


def calc_summary_statistics(
    returns: Union[pd.DataFrame, List],
    annual_factor: int = None,
    provided_excess_returns: bool = True,
    rf: Union[pd.Series, pd.DataFrame] = None,
    var_quantile: Union[float, List] = 0.05,
    timeframes: Union[None, dict] = None,
    return_tangency_weights: bool = True,
    correlations: Union[bool, List] = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
    _timeframe_name: str = None,
):
    """
    Calculates summary statistics for a time series of returns.

    Parameters:
    returns (pd.DataFrame or List): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series or pd.DataFrame, default=None): Risk-free rate data.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    timeframes (dict or None, default=None): Dictionary of timeframes to calculate statistics for each period.
    return_tangency_weights (bool, default=True): If True, returns tangency portfolio weights.
    correlations (bool or list, default=True): If True, returns correlations, or specify columns for correlations.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics of the returns.
    """
    returns = returns.copy()
    if isinstance(rf, (pd.Series, pd.DataFrame)):
        rf = rf.copy()
        if provided_excess_returns is True:
            raise Exception(
                "rf is provided but excess returns were provided as well."
                'Remove "rf" or set "provided_excess_returns" to None or False'
            )

    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(
                series, right_index=True, left_index=True, how="outer"
            )

    if "date" in returns.columns.str.lower():
        returns = returns.rename({"Date": "date"}, axis=1)
        returns = returns.set_index("date")
    returns.index.name = "date"

    try:
        returns.index = pd.to_datetime(returns.index, errors="coerce")
        if returns.index.isnull().any():
            raise ValueError(
                "Index contains invalid datetime values. Ensure the 'returns' index is fully parsable."
            )
    except Exception as e:
        raise ValueError(f"Failed to process the 'date' index: {e}")

    returns = returns.apply(lambda x: x.astype(float))

    if annual_factor is None:
        annual_factor = _calc_annual_factor(list(returns.index))

    if provided_excess_returns is False:
        if rf is not None:
            if len(rf.index) != len(returns.index):
                raise Exception('"rf" index must be the same lenght as "returns"')

    if isinstance(timeframes, dict):
        all_timeframes_summary_statistics = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0] : timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0] :]
            elif timeframe[1]:
                timeframe_returns = returns.loc[: timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f"No returns for {name} timeframe")
            timeframe_returns = timeframe_returns.rename(
                columns=lambda c: c + f" {name}"
            )
            timeframe_summary_statistics = calc_summary_statistics(
                returns=timeframe_returns,
                annual_factor=annual_factor,
                provided_excess_returns=provided_excess_returns,
                rf=rf,
                var_quantile=var_quantile,
                timeframes=None,
                correlations=correlations,
                _timeframe_name=name,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes,
                drop_before_keep=drop_before_keep,
            )
            all_timeframes_summary_statistics = pd.concat(
                [all_timeframes_summary_statistics, timeframe_summary_statistics],
                axis=0,
            )
        return all_timeframes_summary_statistics

    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics["Mean"] = returns.mean()
    summary_statistics["Annualized Mean"] = returns.mean() * annual_factor
    summary_statistics["Vol"] = returns.std()
    summary_statistics["Annualized Vol"] = returns.std() * np.sqrt(annual_factor)
    try:
        if not provided_excess_returns:
            if type(rf) == pd.DataFrame:
                rf = rf.iloc[:, 0].to_list()
            elif type(rf) == pd.Series:
                rf = rf.to_list()
            else:
                raise Exception('"rf" must be either a pd.DataFrame or pd.Series')
            excess_returns = returns.apply(lambda x: x - rf)
            summary_statistics["Sharpe"] = excess_returns.mean() / returns.std()
        else:
            summary_statistics["Sharpe"] = returns.mean() / returns.std()
    except Exception as e:
        print(f"Could not calculate Sharpe: {e}")
    summary_statistics["Annualized Sharpe"] = summary_statistics["Sharpe"] * np.sqrt(
        annual_factor
    )
    summary_statistics["Min"] = returns.min()
    summary_statistics["Max"] = returns.max()
    summary_statistics["Skewness"] = returns.skew()
    summary_statistics["Excess Kurtosis"] = returns.kurtosis()
    var_quantile = (
        [var_quantile] if isinstance(var_quantile, (float, int)) else var_quantile
    )
    for var_q in var_quantile:
        summary_statistics[f"Historical VaR ({var_q:.2%})"] = returns.quantile(
            var_q, axis=0
        )
        summary_statistics[f"Annualized Historical VaR ({var_q:.2%})"] = (
            returns.quantile(var_q, axis=0) * np.sqrt(annual_factor)
        )
        summary_statistics[f"Historical CVaR ({var_q:.2%})"] = returns[
            returns <= returns.quantile(var_q, axis=0)
        ].mean()
        summary_statistics[f"Annualized Historical CVaR ({var_q:.2%})"] = returns[
            returns <= returns.quantile(var_q, axis=0)
        ].mean() * np.sqrt(annual_factor)

    wealth_index = 1000 * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    summary_statistics["Max Drawdown"] = drawdowns.min()
    summary_statistics["Peak"] = [
        previous_peaks[col][: drawdowns[col].idxmin()].idxmax()
        for col in previous_peaks.columns
    ]
    summary_statistics["Bottom"] = drawdowns.idxmin()

    if return_tangency_weights:
        tangency_weights = calc_tangency_weights(returns)
        summary_statistics = summary_statistics.join(tangency_weights)

    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
        recovery_date.append(
            recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        )
    summary_statistics["Recovery"] = recovery_date
    try:
        summary_statistics["Duration (days)"] = [
            (i - j).days if i != "-" else "-"
            for i, j in zip(
                summary_statistics["Recovery"], summary_statistics["Bottom"]
            )
        ]
    except (AttributeError, TypeError) as e:
        print(
            f'Cannot calculate "Drawdown Duration" calculation because there was no recovery or because index are not dates: {str(e)}'
        )

    if correlations is True or isinstance(correlations, list):
        returns_corr = returns.corr()
        if _timeframe_name:
            returns_corr = returns_corr.rename(
                columns=lambda c: c.replace(f" {_timeframe_name}", "")
            )
        returns_corr = returns_corr.rename(columns=lambda c: c + " Correlation")
        if isinstance(correlations, list):
            correlation_names = [c + " Correlation" for c in correlations]
            not_in_returns_corr = [
                c for c in correlation_names if c not in returns_corr.columns
            ]
            if len(not_in_returns_corr) > 0:
                not_in_returns_corr = ", ".join(
                    [c.replace(" Correlation", "") for c in not_in_returns_corr]
                )
                raise Exception(f"{not_in_returns_corr} not in returns columns")
            returns_corr = returns_corr[[c + " Correlation" for c in correlations]]
        summary_statistics = summary_statistics.join(returns_corr)

    if provided_excess_returns is False:
        summary_statistics = summary_statistics.rename(
            {"Sharpe": "Mean / Vol", "Annualized Sharpe": "Annualized Mean / Vol"},
            axis=1,
        )

    return _filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_correlations(
    returns: pd.DataFrame,
    return_only_highest_and_lowest: bool = False,
    matrix_size: Union[int, float, tuple] = 7,
    return_heatmap: bool = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates the correlation matrix of the provided returns and optionally prints or visualizes it.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    print_highest_lowest (bool, default=True): If True, prints the highest and lowest correlations.
    matrix_size (int or float, default=7): Size of the heatmap for correlation matrix visualization.
    return_heatmap (bool, default=True): If True, returns a heatmap of the correlation matrix.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    sns.heatmap or pd.DataFrame: Heatmap of the correlation matrix or the correlation matrix itself.
    """
    returns = returns.copy()

    if "date" in returns.columns.str.lower():
        returns = returns.rename({"Date": "date"}, axis=1)
        returns = returns.set_index("date")
    returns.index.name = "date"

    correlation_matrix = returns.corr()
    if return_heatmap:
        if isinstance(matrix_size, list):
            matrix_size = tuple(matrix_size)
        if isinstance(matrix_size, tuple):
            if len(matrix_size) != 2:
                raise Exception(
                    "matrix_size must be a tuple with two elements (width, height) or a single integer/float"
                )
            figsize = plt.subplots(figsize=matrix_size)
        else:
            figsize = (matrix_size * 1.5, matrix_size)
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = sns.heatmap(
            correlation_matrix,
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            annot=True,
            fmt=".2%",
        )

    if return_only_highest_and_lowest:
        highest_lowest_corr = (
            correlation_matrix.unstack()
            .sort_values()
            .reset_index()
            .set_axis(["asset_1", "asset_2", "corr"], axis=1)
            .loc[lambda df: df.asset_1 != df.asset_2]
        )
        highest_corr = highest_lowest_corr.iloc[lambda df: len(df) - 1, :]
        lowest_corr = highest_lowest_corr.iloc[0, :]
        return pd.DataFrame(
            {
                "First Asset": [highest_corr.asset_1, lowest_corr.asset_1],
                "Second Asset": [highest_corr.asset_2, lowest_corr.asset_2],
                "Correlation": [highest_corr.corr, lowest_corr],
            },
            index=["Highest", "Lowest"],
        )

    if return_heatmap:
        return heatmap
    else:
        return _filter_columns_and_indexes(
            correlation_matrix,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )
