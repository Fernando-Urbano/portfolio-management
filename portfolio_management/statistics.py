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
from portfolio_management.utils import (
    PERIODS_PER_YEAR_MAP,
    clean_returns_df,
    define_periods_per_year,
)

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


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

    Parameters
    ----------
    returns : pd.DataFrame, pd.Series, or list
        Time series of returns.
    calc_positive : bool, optional
        If True, calculates the percentage of positive returns. If False, calculates
        the percentage of negative returns. Defaults to False.
    keep_columns : list or str, optional
        Columns to keep in the resulting DataFrame. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the resulting DataFrame. Defaults to None.
    keep_indexes : list or str, optional
        Indexes to keep in the resulting DataFrame. Defaults to None.
    drop_indexes : list or str, optional
        Indexes to drop from the resulting DataFrame. Defaults to None.
    drop_before_keep : bool, optional
        Whether to drop specified columns/indexes before keeping them. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the percentage of negative or positive returns,
        the total number of returns, and the count of negative or positive returns.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.statistics import calc_negative_pct
    >>> returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    >>> result = calc_negative_pct(returns)
    >>> result
    % Negative Returns       0.5
    Nº Returns              4.0
    Nº Negative Returns     2.0
    Name: 0, dtype: float64
    """
    returns = clean_returns_df(returns)
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

    Parameters
    ----------
    returns : pd.DataFrame or pd.Series
        Time series of returns.
    return_plot : bool, optional
        If True, plots the cumulative returns. Defaults to True.
    fig_size : tuple, optional
        Size of the plot (width, height). Defaults to (7, 5).
    return_series : bool, optional
        If True, returns the cumulative returns as a DataFrame instead of None. Defaults to False.
    name : str, optional
        Name for the plot title or the returned series. Defaults to None.
    timeframes : dict or None, optional
        Dictionary of timeframes (key = label, value = (start, end)) for which to calculate
        and optionally plot cumulative returns separately. Defaults to None.

    Returns
    -------
    pd.DataFrame or None
        If `return_series` is True, returns a DataFrame of cumulative returns.
        Otherwise, returns None.

    Raises
    ------
    Exception
        If a specified timeframe has no data in `returns`.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.statistics import calc_cumulative_returns
    >>> returns_df = pd.DataFrame({
    ...     'AssetA': [0.01, -0.02, 0.03],
    ...     'AssetB': [0.005, 0.007, -0.002]
    ... })
    >>> # Plot cumulative returns and also return the series:
    >>> cum_df = calc_cumulative_returns(returns_df, return_plot=True, return_series=True)
    >>> cum_df.head()
    """
    returns = clean_returns_df(returns)
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

    Parameters
    ----------
    summary_statistics : pd.DataFrame
        DataFrame containing summary statistics for each asset (rows) and various metrics (columns).
    stat : str, optional
        The statistic (column name) to compare assets by. Defaults to 'Annualized Sharpe'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows of the best and worst assets according to the specified statistic.

    Raises
    ------
    Exception
        If `summary_statistics` has fewer than two rows.
    ValueError
        If `stat` is not found in the columns of `summary_statistics`.
    ValueError
        If all values in the specified `stat` column are missing.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.statistics import get_best_and_worst
    >>> data = {
    ...     'Annualized Sharpe': [1.2, 0.8, 0.5],
    ...     'Another Stat': [0.1, 0.2, 0.3]
    ... }
    >>> index = ['AssetA', 'AssetB', 'AssetC']
    >>> summary_df = pd.DataFrame(data, index=index)
    >>> get_best_and_worst(summary_df, stat='Annualized Sharpe')
        Annualized Sharpe  Another Stat
    AssetA               1.2           0.1
    AssetC               0.5           0.3
    """
    summary_statistics = summary_statistics.copy()

    if len(summary_statistics.index) < 2:
        raise Exception(
            '"summary_statistics" must have at least two lines in order to do comparison'
        )

    if stat not in summary_statistics.columns:
        raise ValueError(f'{stat} not in "summary_statistics"')
    summary_statistics.rename(columns=lambda c: c.replace(" ", "").lower())
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
    periods_per_year: int = None,
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

    Parameters
    ----------
    returns : pd.DataFrame or list
        Time series of returns. Columns represent assets or factors.
    periods_per_year : int, optional
        Number of periods per year for annualizing returns. If None, attempts to infer from the data.
        Defaults to None.
    provided_excess_returns : bool, optional
        If True, indicates that `returns` are already excess returns (i.e., returns - risk-free rate).
        If False or None, will subtract `rf` (if provided). Defaults to True.
    rf : pd.Series or pd.DataFrame, optional
        Risk-free rate data with the same index as `returns`. Only used if `provided_excess_returns` is False.
        Defaults to None.
    var_quantile : float or list of float, optional
        Quantile(s) for calculating Value at Risk (VaR) and Conditional VaR (CVaR). Defaults to 0.05.
    timeframes : dict or None, optional
        Dictionary of timeframes (key = label, value = (start, end)) for which to calculate
        summary statistics separately. Defaults to None.
    return_tangency_weights : bool, optional
        If True, includes tangency portfolio weights in the summary. Defaults to True.
    correlations : bool or list, optional
        If True, appends the correlation matrix to the summary. If a list of column names, appends
        correlations only for those columns. Defaults to True.
    keep_columns : list or str, optional
        Columns to keep in the final DataFrame. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the final DataFrame. Defaults to None.
    keep_indexes : list or str, optional
        Indexes (rows) to keep in the final DataFrame. Defaults to None.
    drop_indexes : list or str, optional
        Indexes (rows) to drop from the final DataFrame. Defaults to None.
    drop_before_keep : bool, optional
        Whether to drop specified columns/indexes before keeping. Defaults to False.
    _timeframe_name : str, optional
        Internal parameter for naming a timeframe. Not typically set by the user. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing summary statistics such as mean, volatility, Sharpe ratios, VaR, CVaR,
        drawdowns, tangency weights (optional), and correlations (optional).

    Raises
    ------
    Exception
        If both `rf` is provided and `provided_excess_returns` is True.
    Exception
        If `rf` does not match the index length of `returns` when `provided_excess_returns` is False.
    Exception
        If a specified timeframe has no data in `returns`.
    ValueError
        If the columns specified for correlation do not exist in `returns`.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.statistics import calc_summary_statistics
    >>> returns_df = pd.DataFrame({
    ...     'AssetA': [0.01, -0.02, 0.03],
    ...     'AssetB': [0.005, 0.007, -0.002]
    ... })
    >>> stats_df = calc_summary_statistics(returns_df)
    >>> stats_df.head()
    """
    returns = returns.copy()
    if isinstance(rf, (pd.Series, pd.DataFrame)):
        rf = rf.copy()
        if provided_excess_returns is True:
            raise Exception(
                "rf is provided but excess returns were provided as well."
                'Remove "rf" or set "provided_excess_returns" to None or False'
            )

    returns = clean_returns_df(returns)
    periods_per_year = define_periods_per_year(returns, periods_per_year)

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
                periods_per_year=periods_per_year,
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
    summary_statistics["Annualized Mean"] = returns.mean() * periods_per_year
    summary_statistics["Vol"] = returns.std()
    summary_statistics["Annualized Vol"] = returns.std() * np.sqrt(periods_per_year)
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
        periods_per_year
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
            returns.quantile(var_q, axis=0) * np.sqrt(periods_per_year)
        )
        summary_statistics[f"Historical CVaR ({var_q:.2%})"] = returns[
            returns <= returns.quantile(var_q, axis=0)
        ].mean()
        summary_statistics[f"Annualized Historical CVaR ({var_q:.2%})"] = returns[
            returns <= returns.quantile(var_q, axis=0)
        ].mean() * np.sqrt(periods_per_year)

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
    Calculates the correlation matrix of the provided returns and optionally visualizes it.

    Parameters
    ----------
    returns : pd.DataFrame
        Time series of returns.
    return_only_highest_and_lowest : bool, optional
        If True, returns a DataFrame with only the highest and lowest correlations. Defaults to False.
    matrix_size : int, float, or tuple, optional
        Size for the heatmap figure. If a single int/float, uses (matrix_size * 1.5, matrix_size).
        If a tuple of length 2, uses that as (width, height). Defaults to 7.
    return_heatmap : bool, optional
        If True, returns a seaborn heatmap object of the correlation matrix. If False,
        returns the correlation matrix as a DataFrame. Defaults to True.
    keep_columns : list or str, optional
        Columns to keep in the resulting DataFrame. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the resulting DataFrame. Defaults to None.
    keep_indexes : list or str, optional
        Indexes to keep in the resulting DataFrame. Defaults to None.
    drop_indexes : list or str, optional
        Indexes to drop from the resulting DataFrame. Defaults to None.
    drop_before_keep : bool, optional
        Whether to drop specified columns/indexes before keeping. Defaults to False.

    Returns
    -------
    sns.heatmap or pd.DataFrame
        - If `return_only_highest_and_lowest` is True, returns a DataFrame with the highest and lowest correlations.
        - Else if `return_heatmap` is True, returns a seaborn heatmap object.
        - Otherwise, returns a DataFrame of the full correlation matrix.

    Raises
    ------
    Exception
        If `matrix_size` is a tuple with a length other than 2.
    Exception
        If specified columns or indexes to keep/drop are not found in `returns`.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.statistics import calc_correlations
    >>> returns_df = pd.DataFrame({
    ...     'AssetA': [0.01, 0.02, -0.01],
    ...     'AssetB': [0.005, 0.007, 0.002]
    ... })
    >>> # Get a heatmap:
    >>> heatmap = calc_correlations(returns_df, return_heatmap=True)
    >>> # Or get the correlation DataFrame:
    >>> corr_df = calc_correlations(returns_df, return_heatmap=False)
    >>> corr_df
    """
    returns = clean_returns_df(returns)

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
