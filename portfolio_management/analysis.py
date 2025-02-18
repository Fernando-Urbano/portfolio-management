import pandas as pd
import numpy as np
import re
import math
import datetime
from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from warnings import warn, filterwarnings
from arch import arch_model
from collections import defaultdict
from scipy.stats import norm
from portfolio_management.port_construction import calc_tangency_weights
from portfolio_management.utils import (
    _filter_columns_and_indexes,
    clean_returns_df,
    define_periods_per_year,
)

pd.options.display.float_format = "{:,.4f}".format
filterwarnings("ignore")


def calc_cross_section_regression(
    returns: Union[pd.DataFrame, List],
    factors: Union[pd.DataFrame, List],
    periods_per_year: int = None,
    provided_excess_returns: bool = None,
    rf: pd.Series = None,
    return_model: bool = False,
    name: str = None,
    return_mae: bool = True,
    intercept_cross_section: bool = True,
    return_historical_premium: bool = True,
    return_annualized_premium: bool = True,
    compare_premiums: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Performs a cross-sectional regression on the provided returns and factors.

    Parameters
    ----------
    returns : pd.DataFrame or list
        Time series of asset returns.
    factors : pd.DataFrame or list
        Time series of factor data.
    periods_per_year : int, optional
        Number of periods per year for annualizing returns. If None, attempts to infer from `returns`.
        Defaults to None.
    provided_excess_returns : bool, optional
        Whether `returns` are already excess returns (i.e., returns minus risk-free rate). If None,
        a warning is issued and it defaults to True. Defaults to None.
    rf : pd.Series, optional
        Risk-free rate data, used if `provided_excess_returns` is False. Must have the same index as `returns`.
        Defaults to None.
    return_model : bool, optional
        If True, returns the fitted regression model instead of a DataFrame of results. Defaults to False.
    name : str, optional
        Name/label for the regression. Defaults to None.
    return_mae : bool, optional
        If True, calculates and returns mean absolute error (MAE) metrics. Defaults to True.
    intercept_cross_section : bool, optional
        If True, includes an intercept in the cross-sectional regression. Defaults to True.
    return_historical_premium : bool, optional
        If True, adds columns for the average (historical) factor premiums. Defaults to True.
    return_annualized_premium : bool, optional
        If True, adds columns for the annualized factor premiums. Defaults to True.
    compare_premiums : bool, optional
        If True, compares historical vs. estimated premiums in a single table. Defaults to False.
    keep_columns : list or str, optional
        Columns to keep in the final output. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the final output. Defaults to None.
    keep_indexes : list or str, optional
        Indexes (rows) to keep in the final output. Defaults to None.
    drop_indexes : list or str, optional
        Indexes (rows) to drop from the final output. Defaults to None.
    drop_before_keep : bool, optional
        If True, drops specified columns/indexes before keeping the specified ones. Defaults to False.

    Returns
    -------
    pd.DataFrame or statsmodels.regression.linear_model.RegressionResults
        - If `return_model` is False, returns a DataFrame containing cross-sectional regression results.
        - If `return_model` is True, returns the fitted OLS model object.

    Raises
    ------
    Exception
        If `rf` is provided but its index length does not match `returns` when `provided_excess_returns` is False.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.analysis import calc_cross_section_regression
    >>> returns_df = pd.DataFrame({
    ...     'AssetA': [0.01, -0.02, 0.03],
    ...     'AssetB': [0.005, 0.007, -0.002]
    ... })
    >>> factors_df = pd.DataFrame({
    ...     'Factor1': [0.001, 0.002, 0.003],
    ...     'Factor2': [0.0005, 0.0007, 0.001]
    ... })
    >>> # Run a cross-sectional regression and get a summary DataFrame:
    >>> result = calc_cross_section_regression(returns_df, factors_df)
    >>> result
    """
    returns = clean_returns_df(returns)
    factors = clean_returns_df(factors)
    periods_per_year = define_periods_per_year(returns, periods_per_year)

    if isinstance(rf, (pd.Series, pd.DataFrame)):
        rf = rf.copy()

    if compare_premiums:
        return_historical_premium = True
        return_annualized_premium = True

    if provided_excess_returns is None:
        warn(
            "Assuming excess returns were provided. Set 'provided_excess_returns' to silence this warning"
        )
        provided_excess_returns = True
    elif provided_excess_returns is False:
        if rf is not None:
            if len(rf.index) != len(returns.index):
                raise Exception('"rf" index must be the same lenght as "returns"')
            returns = returns.sub(rf.values, axis=0)

    time_series_regressions = calc_iterative_regression(
        returns, factors, periods_per_year=periods_per_year, warnings=False
    )
    time_series_betas = time_series_regressions.filter(regex="Beta$", axis=1)
    time_series_historical_returns = time_series_regressions[["Fitted Mean"]]
    cross_section_regression = calc_regression(
        time_series_historical_returns,
        time_series_betas,
        periods_per_year=periods_per_year,
        intercept=intercept_cross_section,
        return_model=return_model,
        warnings=False,
    )

    if return_model:
        return cross_section_regression
    cross_section_regression = cross_section_regression.rename(
        columns=lambda c: c.replace(" Beta Beta", " Lambda").replace("Alpha", "Eta")
    )
    if name is None:
        name = " + ".join(
            [
                c.replace(" Lambda", "")
                for c in cross_section_regression.filter(
                    regex=" Lambda$", axis=1
                ).columns
            ]
        )
    cross_section_regression.index = [f"{name} Cross-Section Regression"]
    cross_section_regression.drop(
        [
            "Information Ratio",
            "Annualized Information Ratio",
            "Tracking Error",
            "Annualized Tracking Error",
            "Fitted Mean",
            "Annualized Fitted Mean",
        ],
        axis=1,
        inplace=True,
    )
    if return_annualized_premium:
        factors_annualized_premium = (
            cross_section_regression.filter(regex=" Lambda$", axis=1)
            .apply(lambda x: x * periods_per_year)
            .rename(columns=lambda c: c.replace(" Lambda", " Annualized Lambda"))
        )
        cross_section_regression = cross_section_regression.join(
            factors_annualized_premium
        )

    if return_historical_premium:
        factors_historical_premium = (
            factors.mean()
            .to_frame(f"{name} Cross-Section Regression")
            .transpose()
            .rename(columns=lambda c: c + " Historical Premium")
        )
        cross_section_regression = cross_section_regression.join(
            factors_historical_premium
        )
        if return_annualized_premium:
            factors_annualized_historical_premium = factors_historical_premium.apply(
                lambda x: x * periods_per_year
            ).rename(
                columns=lambda c: c.replace(
                    " Historical Premium", " Annualized Historical Premium"
                )
            )
            cross_section_regression = cross_section_regression.join(
                factors_annualized_historical_premium
            )

    if compare_premiums:
        cross_section_regression = cross_section_regression.filter(
            regex="Lambda$|Historical Premium$", axis=1
        )
        cross_section_regression = cross_section_regression.transpose()
        factor_index = cross_section_regression.index.str.extract(
            f'({"|".join(list(factors.columns))})'
        ).values
        cross_section_regression["Factor"] = [
            f[0] if isinstance(f, (list, tuple, np.ndarray, np.array)) else f
            for f in factor_index
        ]
        cross_section_regression["Premium Type"] = (
            cross_section_regression.index.str.replace(
                f'({"|".join(list(factors.columns))})', ""
            )
        )
        premiums_comparison = cross_section_regression.pivot(
            index="Factor",
            columns="Premium Type",
            values=f"{name} Cross-Section Regression",
        )
        premiums_comparison.columns.name = None
        premiums_comparison.index.name = None
        premiums_comparison.join(calc_tangency_weights(factors))
        premiums_comparison = premiums_comparison.join(
            factors.corr().rename(columns=lambda c: c + " Correlation")
        )
        return _filter_columns_and_indexes(
            premiums_comparison,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )

    if return_mae:
        cross_section_regression["TS MAE"] = (
            time_series_regressions["Alpha"].abs().mean()
        )
        cross_section_regression["TS Annualized MAE"] = (
            time_series_regressions["Annualized Alpha"].abs().mean()
        )
        cross_section_regression_model = calc_regression(
            time_series_historical_returns,
            time_series_betas,
            periods_per_year=periods_per_year,
            intercept=intercept_cross_section,
            return_model=True,
            warnings=False,
        )
        cross_section_regression["CS MAE"] = (
            cross_section_regression_model.resid.abs().mean()
        )
        cross_section_regression["CS Annualized MAE"] = (
            cross_section_regression["CS MAE"] * periods_per_year
        )

    return _filter_columns_and_indexes(
        cross_section_regression,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_regression(
    y: Union[pd.DataFrame, pd.Series],
    X: Union[pd.DataFrame, pd.Series],
    intercept: bool = True,
    periods_per_year: Union[None, int] = None,
    warnings: bool = True,
    return_model: bool = False,
    return_fitted_values: bool = False,
    name_fitted_values: str = None,
    calc_treynor_info_ratios: bool = True,
    timeframes: Union[None, dict] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
    calc_sortino_ratio: bool = False,
    is_time_series_regression: bool = False,
):
    """
    Performs an OLS regression on the provided data with optional intercept, timeframes, and statistical ratios.

    Parameters
    ----------
    y : pd.DataFrame or pd.Series
        Dependent variable(s) for the regression (one column if DataFrame).
    X : pd.DataFrame or pd.Series
        Independent variable(s) for the regression. If multiple columns, each is treated as a separate predictor.
    intercept : bool, optional
        If True, includes an intercept in the regression. Defaults to True.
    periods_per_year : int or None, optional
        Number of periods per year for annualizing regression statistics. If None and
        `is_time_series_regression` is False, defaults to 12. Defaults to None.
    warnings : bool, optional
        If True, prints warnings (e.g., about index alignment). Defaults to True.
    return_model : bool, optional
        If True, returns the fitted regression model object (from `statsmodels`). Defaults to False.
    return_fitted_values : bool, optional
        If True, returns a DataFrame of fitted values instead of summary statistics. Implies `return_model=True`.
        Defaults to False.
    name_fitted_values : str, optional
        Name for the fitted values column if `return_fitted_values` is True. Defaults to None.
    calc_treynor_info_ratios : bool, optional
        If True, calculates Treynor and Information ratios (only valid if there's exactly one factor
        when including an intercept). Defaults to True.
    timeframes : dict or None, optional
        Dictionary of timeframes (key = label, value = (start, end)) for which to run separate regressions.
        Defaults to None.
    keep_columns : list or str, optional
        Columns to keep in the final output. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the final output. Defaults to None.
    keep_indexes : list or str, optional
        Indexes (rows) to keep in the final output. Defaults to None.
    drop_indexes : list or str, optional
        Indexes (rows) to drop from the final output. Defaults to None.
    drop_before_keep : bool, optional
        If True, drops specified columns/indexes before keeping the specified ones. Defaults to False.
    calc_sortino_ratio : bool, optional
        If True, calculates the Sortino ratio. Defaults to False.
    is_time_series_regression : bool, optional
        If True, attempts to infer `periods_per_year` from the data (e.g., daily, monthly). Defaults to False.

    Returns
    -------
    pd.DataFrame or statsmodels.regression.linear_model.RegressionResults
        - If `return_model` is False and `return_fitted_values` is False, returns a DataFrame of regression summary statistics.
        - If `return_model` is True but `return_fitted_values` is False, returns the fitted OLS model.
        - If `return_fitted_values` is True, returns a DataFrame of fitted values.

    Raises
    ------
    Exception
        If `y` has more than one column.
    ValueError
        If `periods_per_year` is not a positive integer when provided.
    Exception
        If `y` and `X` have mismatched indexes that result in fewer than 4 observations after alignment.
    Exception
        If no data is found for a given timeframe in `timeframes`.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.analysis import calc_regression
    >>> y = pd.Series([0.01, 0.02, -0.01], name='AssetA')
    >>> X = pd.DataFrame({'Factor1': [0.001, 0.002, 0.003]})
    >>> # Simple regression with intercept:
    >>> result_df = calc_regression(y, X, intercept=True)
    >>> result_df
    """
    y = y.copy()
    X = X.copy()

    y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
    X_names = " + ".join(list(X.columns))
    X_names = "Intercept + " + X_names if intercept else X_names

    return_model = return_model if not return_fitted_values else True

    if is_time_series_regression:
        periods_per_year = define_periods_per_year(
            clean_returns_df(y), periods_per_year
        )
    elif periods_per_year is None:
        periods_per_year = 12
        warn(
            "Assuming 12 periods per year. Set 'periods_per_year' to silence "
            + "this warning or speficy 'is_time_series_regression' as True to "
            + "allow for estimate of 'periods_per_year'"
        )
    elif isinstance(periods_per_year, (float)):
        if periods_per_year % 1 != 0:
            raise ValueError("Periods per year must be an integer")
        periods_per_year = int(periods_per_year)
    if isinstance(periods_per_year, (int)):
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer")

    if intercept:
        X = sm.add_constant(X)

    y_name = y.name if isinstance(y, pd.Series) else y.columns[0]

    if isinstance(y, pd.Series):
        y = y.to_frame(y_name)

    if y.shape[1] > 1:
        raise Exception(
            f"y has more than one column. Please provide a single column DataFrame or Series"
        )

    if len(X.index) != len(y.index):
        print(
            f"y has lenght {len(y.index)} and X has lenght {len(X.index)}. Joining y and X by index..."
        )
        df = y.join(X, how="left")
        df = df.dropna()
        y = df[y_name]
        X = df.drop(y_name, axis=1)
        if len(X.index) < 4:
            raise Exception(
                "Indexes of y and X do not match and there are less than 4 observations. Cannot calculate regression"
            )

    if isinstance(timeframes, dict):
        all_timeframes_regressions = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_y = y.loc[timeframe[0] : timeframe[1]]
                timeframe_X = X.loc[timeframe[0] : timeframe[1]]
            elif timeframe[0]:
                timeframe_y = y.loc[timeframe[0] :]
                timeframe_X = X.loc[timeframe[0] :]
            elif timeframe[1]:
                timeframe_y = y.loc[: timeframe[1]]
                timeframe_X = X.loc[: timeframe[1]]
            else:
                timeframe_y = y.copy()
                timeframe_X = X.copy()
            if len(timeframe_y.index) == 0 or len(timeframe_X.index) == 0:
                raise Exception(f"No returns for {name} timeframe")
            timeframe_regression = calc_regression(
                y=timeframe_y,
                X=timeframe_X,
                intercept=intercept,
                periods_per_year=periods_per_year,
                warnings=False,
                return_model=False,
                calc_treynor_info_ratios=calc_treynor_info_ratios,
                timeframes=None,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes,
                drop_before_keep=drop_before_keep,
            )
            timeframe_regression.index = [timeframe_regression.index + " " + name]
            all_timeframes_regressions = pd.concat(
                [all_timeframes_regressions, timeframe_regression], axis=0
            )
        return all_timeframes_regressions

    try:
        model = sm.OLS(y, X, missing="drop", hasconst=intercept)
    except ValueError:
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = sm.OLS(y, X, missing="drop", hasconst=intercept)
        if warnings:
            warn(
                f'"{y_name}" Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame'
            )
    results = model.fit()
    summary = dict()

    if return_model:
        if not return_fitted_values:
            return results
        else:
            fitted_values = results.fittedvalues
            if name_fitted_values is None:
                name_fitted_values = f"{y_name} ~ {X_names}"
            fitted_values = fitted_values.to_frame(name_fitted_values)
            return fitted_values

    inter = results.params[0] if intercept else None
    betas = results.params[1:] if intercept else results.params

    summary["Alpha"] = inter if inter is not None else "-"
    summary["Annualized Alpha"] = inter * periods_per_year if inter is not None else "-"
    summary["R-Squared"] = results.rsquared

    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)

    X_assets = X.columns[1:] if intercept else X.columns
    for i, asset_name in enumerate(X_assets):
        summary[f"{asset_name} Beta"] = betas[i]

    if calc_treynor_info_ratios:
        if len([c for c in X.columns if c != "const"]) == 1:
            summary["Treynor Ratio"] = y.mean() / betas[0]
            summary["Annualized Treynor Ratio"] = (
                summary["Treynor Ratio"] * periods_per_year
            )
        summary["Information Ratio"] = (
            (inter / results.resid.std()) if intercept else "-"
        )
        summary["Annualized Information Ratio"] = (
            summary["Information Ratio"] * np.sqrt(periods_per_year)
            if intercept
            else "-"
        )
    summary["Tracking Error"] = results.resid.std()
    summary["Annualized Tracking Error"] = results.resid.std() * np.sqrt(
        periods_per_year
    )
    summary["Fitted Mean"] = results.fittedvalues.mean()
    summary["Annualized Fitted Mean"] = summary["Fitted Mean"] * periods_per_year
    if calc_sortino_ratio:
        try:
            summary["Sortino Ratio"] = summary["Fitted Mean"] / y[y < 0].std()
            summary["Annualized Sortino Ratio"] = summary["Sortino Ratio"] * np.sqrt(
                periods_per_year
            )
        except Exception as e:
            print(
                f'Cannot calculate Sortino Ratio: {str(e)}. Set "calc_sortino_ratio" to False or review function'
            )
    y_name = f"{y_name} no Intercept" if not intercept else y_name
    return _filter_columns_and_indexes(
        pd.DataFrame(summary, index=[y_name]),
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_iterative_regression(
    multiple_y: Union[pd.DataFrame, pd.Series],
    X: Union[pd.DataFrame, pd.Series],
    periods_per_year: Union[None, int] = None,
    intercept: bool = True,
    warnings: bool = True,
    calc_treynor_info_ratios: bool = True,
    calc_sortino_ratio: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Performs iterative regression across multiple dependent variables (assets), each regressed on the same predictors.

    Parameters
    ----------
    multiple_y : pd.DataFrame or pd.Series
        Dependent variables (assets). If multiple columns, each column is treated as a separate dependent variable.
    X : pd.DataFrame or pd.Series
        Independent variable(s) (predictors) used for all assets in `multiple_y`.
    periods_per_year : int or None, optional
        Number of periods per year for annualizing statistics. If None, attempts to infer from the data. Defaults to 12.
    intercept : bool, optional
        If True, includes an intercept in each regression. Defaults to True.
    warnings : bool, optional
        If True, prints warnings (e.g., about index alignment). Defaults to True.
    calc_treynor_info_ratios : bool, optional
        If True, calculates Treynor and Information ratios (when applicable). Defaults to True.
    calc_sortino_ratio : bool, optional
        If True, calculates the Sortino ratio. Defaults to False.
    keep_columns : list or str, optional
        Columns to keep in the final output. Defaults to None.
    drop_columns : list or str, optional
        Columns to drop from the final output. Defaults to None.
    keep_indexes : list or str, optional
        Indexes (rows) to keep in the final output. Defaults to None.
    drop_indexes : list or str, optional
        Indexes (rows) to drop from the final output. Defaults to None.
    drop_before_keep : bool, optional
        If True, drops specified columns/indexes before keeping the specified ones. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing summary statistics for each asset's regression.

    Raises
    ------
    Exception
        If a timeframe is specified with no matching data (when `define_periods_per_year` is used or if the user-specified
        timeframe has no overlap with the data).

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.analysis import calc_iterative_regression
    >>> returns_df = pd.DataFrame({
    ...     'AssetA': [0.01, 0.02, -0.01],
    ...     'AssetB': [0.005, -0.003, 0.004]
    ... })
    >>> factor_df = pd.DataFrame({'Factor1': [0.001, 0.002, 0.003]})
    >>> # Run iterative regression for each asset against the same factor(s):
    >>> iterative_results = calc_iterative_regression(returns_df, factor_df)
    >>> iterative_results
    """
    multiple_y = clean_returns_df(multiple_y)
    X = clean_returns_df(X)
    periods_per_year = define_periods_per_year(multiple_y, periods_per_year)

    regressions = pd.DataFrame({})
    for asset in multiple_y.columns:
        y = multiple_y[[asset]]
        new_regression = calc_regression(
            y,
            X,
            periods_per_year=periods_per_year,
            intercept=intercept,
            warnings=warnings,
            calc_treynor_info_ratios=calc_treynor_info_ratios,
            calc_sortino_ratio=calc_sortino_ratio,
        )
        warnings = False
        regressions = pd.concat([regressions, new_regression], axis=0)

    return _filter_columns_and_indexes(
        regressions,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )
