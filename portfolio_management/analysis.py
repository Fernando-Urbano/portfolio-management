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
from portfolio_management.port_construction import calc_tangency_weights
from portfolio_management.utils import (
    _filter_columns_and_indexes,
    clean_returns_df,
    define_periods_per_year,
)

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")


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

    Parameters:
    returns (pd.DataFrame or list): Time series of returns.
    factors (pd.DataFrame or list): Time series of factor data.
    periods_per_year (int, default=None): Factor for annualizing returns.
    provided_excess_returns (bool, default=None): Whether excess returns are already provided.
    rf (pd.Series, default=None): Risk-free rate data for subtracting from returns.
    return_model (bool, default=False): If True, returns the regression model.
    name (str, default=None): Name for labeling the regression.
    return_mae (bool, default=True): If True, returns the mean absolute error of the regression.
    intercept_cross_section (bool, default=True): If True, includes an intercept in the cross-sectional regression.
    return_historical_premium (bool, default=True): If True, returns the historical premium of factors.
    return_annualized_premium (bool, default=True): If True, returns the annualized premium of factors.
    compare_premiums (bool, default=False): If True, compares the historical and estimated premiums.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame or model: Cross-sectional regression output or the model if `return_model` is True.
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
        warnings.warn(
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
        cross_section_regression["Factor"] = cross_section_regression.index.str.extract(
            f'({"|".join(list(factors.columns))})'
        ).values
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
):
    """
    Performs an OLS regression on the provided data with optional intercept, timeframes, and statistical ratios.

    Parameters:
    y (pd.DataFrame or pd.Series): Dependent variable for the regression.
    X (pd.DataFrame or pd.Series): Independent variable(s) for the regression.
    intercept (bool, default=True): If True, includes an intercept in the regression.
    periods_per_year (int or None, default=None): Factor for annualizing regression statistics.
    warnings (bool, default=True): If True, prints warnings about assumptions.
    return_model (bool, default=False): If True, returns the regression model object.
    return_fitted_values (bool, default=False): If True, returns the fitted values of the regression.
    name_fitted_values (str, default=None): Name for the fitted values column.
    calc_treynor_info_ratios (bool, default=True): If True, calculates Treynor and Information ratios.
    timeframes (dict or None, default=None): Dictionary of timeframes to run separate regressions for each period.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.
    calc_sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.

    Returns:
    pd.DataFrame or model: Regression summary statistics or the model if `return_model` is True.
    """
    y = y.copy()
    X = X.copy()

    y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
    X_names = " + ".join(list(X.columns))
    X_names = "Intercept + " + X_names if intercept else X_names

    return_model = return_model if not return_fitted_values else True

    periods_per_year = define_periods_per_year(clean_returns_df(y), periods_per_year)

    X = clean_returns_df(X)

    if warnings:
        warnings.warn(
            '"calc_regression" assumes excess returns to calculate Information and Treynor Ratios'
        )
    if intercept:
        X = sm.add_constant(X)

    y_name = y.name if isinstance(y, pd.Series) else y.columns[0]

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
            print(
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
    Performs iterative regression across multiple dependent variables (assets).

    Parameters:
    multiple_y (pd.DataFrame or pd.Series): Dependent variables for multiple assets.
    X (pd.DataFrame or pd.Series): Independent variable(s) (predictors).
    periods_per_year (int or None, default=12): Factor for annualizing regression statistics.
    intercept (bool, default=True): If True, includes an intercept in the regression.
    warnings (bool, default=True): If True, prints warnings about assumptions.
    calc_treynor_info_ratios (bool, default=True): If True, calculates Treynor and Information ratios.
    calc_sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics for each asset regression.
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
