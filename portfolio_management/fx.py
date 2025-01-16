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


def calc_fx_exc_ret(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    periods_per_year: Union[int, None] = None,
    return_exc_ret: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates foreign exchange excess returns by subtracting risk-free rates from FX rates.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    periods_per_year (int or None, default=None): Factor for annualizing the returns.
    return_exc_ret (bool, default=False): If True, returns the excess returns instead of summary statistics.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary statistics or excess returns based on FX rates and risk-free rates.
    """
    raise Exception("Function not available - needs testing prior to use")
    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {"GBP1M": "USUK", "EUR1M": "USEU", "CHF1M": "USSZ", "JPY1M": "USJP"}

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print(
            "No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate."
        )
        base_rf = "USD1M"
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    all_fx_holdings_exc_ret = pd.DataFrame({})
    for rf, fx in rf_to_fx.items():
        fx_holdings_exc_ret = (
            fx_rates[fx]
            - fx_rates[fx].shift(1)
            + rf_rates[rf].shift(1)
            - base_rf_series.shift(1)
        )
        try:
            rf_name = re.sub("[0-9]+M", "", rf)
        except:
            rf_name = rf
        fx_holdings_exc_ret = fx_holdings_exc_ret.dropna(axis=0).to_frame(rf_name)
        all_fx_holdings_exc_ret = all_fx_holdings_exc_ret.join(
            fx_holdings_exc_ret, how="outer"
        )

    if not return_exc_ret:
        return _filter_columns_and_indexes(
            calc_summary_statistics(
                all_fx_holdings_exc_ret, periods_per_year=periods_per_year
            ),
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )
    else:
        return _filter_columns_and_indexes(
            all_fx_holdings_exc_ret,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )


def check_if_uip_holds(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    provided_log_fx_rates: bool = False,
    provided_log_rf_rates: bool = False,
    rf_to_fx: dict = None,
    base_rf: str = None,
    periods_per_year: Union[int, None] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    # TODO
    pass


def calc_fx_regression(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    periods_per_year: Union[int, None] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
    print_analysis: bool = True,
):
    """
    Calculates FX regression and provides an analysis of how the risk-free rate differentials affect FX rates.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    periods_per_year (int or None, default=None): Factor for annualizing returns.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.
    print_analysis (bool, default=True): If True, prints an analysis of the regression results.

    Returns:
    pd.DataFrame: Summary of regression statistics for the FX rates and risk-free rate differentials.
    """
    raise Exception("Function not available - needs testing prior to use")
    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {"GBP1M": "USUK", "EUR1M": "USEU", "CHF1M": "USSZ", "JPY1M": "USJP"}

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print(
            "No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate."
        )
        base_rf = "USD1M"
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    if periods_per_year is None:
        print(
            "Regression assumes 'periods_per_year' equals to 12 since it was not provided"
        )
        periods_per_year = 12

    all_regressions_summary = pd.DataFrame({})

    for rf, fx in rf_to_fx.items():
        try:
            rf_name = re.sub("[0-9]+M", "", rf)
        except:
            rf_name = rf
        factor = (base_rf_series - rf_rates[rf]).to_frame("Base RF - Foreign RF")
        strat = fx_rates[fx].diff().to_frame(rf_name)
        regression_summary = calc_regression(
            strat, factor, periods_per_year=periods_per_year, warnings=False
        )
        all_regressions_summary = pd.concat(
            [all_regressions_summary, regression_summary]
        )

    if print_analysis:
        try:
            print("\n" * 2)
            for currency in all_regressions_summary.index:
                fx_beta = all_regressions_summary.loc[
                    currency, "Base RF - Foreign RF Beta"
                ]
                fx_alpha = all_regressions_summary.loc[currency, "Alpha"]
                print(
                    f"For {currency} against the base currency, the Beta is {fx_beta:.2f}."
                )
                if 1.1 >= fx_beta and fx_beta >= 0.85:
                    print(
                        "which shows that, on average, the difference in risk-free rate is mainly offset by the FX rate movement."
                    )
                elif fx_beta > 1.1:
                    print(
                        "which shows that, on average, the difference in risk-free rate is more than offset by the FX rate movement.,\n"
                        "Therefore, on average, the currency with the lower risk-free rate outperforms."
                    )
                elif fx_beta < 0.85 and fx_beta > 0.15:
                    print(
                        "which shows that, on average, the difference in risk-free rate is only partially offset by the FX rate movement.\n"
                        "Therefore, on average, the currency with the higher risk-free rate outperforms."
                    )
                elif fx_beta <= 0.15 and fx_beta >= -0.1:
                    print(
                        "which shows that, on average, the difference in risk-free rate is almost not offset by the FX rate movement.\n"
                        "Therefore, on average, the currency with the higher risk-free rate outperforms."
                    )
                elif fx_beta <= 0.15 and fx_beta >= -0.1:
                    print(
                        "which shows that, on average, the difference in risk-free rate is almost not offset by the FX rate movement.\n"
                        "Therefore, on average, the currency with the higher risk-free rate outperforms."
                    )
                else:
                    print(
                        "which shows that, on average, the change FX rate helps the currency with the highest risk-free return.\n"
                        "Therefore, the difference between returns is increased, on average, by the changes in the FX rate."
                    )
                print("\n" * 2)
        except:
            print("Could not print analysis. Review function.")

    return _filter_columns_and_indexes(
        all_regressions_summary,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )


def calc_dynamic_carry_trade(
    fx_rates: pd.DataFrame,
    rf_rates: pd.DataFrame,
    transform_to_log_fx_rates: bool = True,
    transform_to_log_rf_rates: bool = True,
    rf_to_fx: dict = None,
    base_rf: str = None,
    base_rf_series: Union[pd.Series, pd.DataFrame] = None,
    periods_per_year: Union[int, None] = None,
    return_premium_series: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    drop_before_keep: bool = False,
):
    """
    Calculates the dynamic carry trade strategy based on FX rates and risk-free rate differentials.

    Parameters:
    fx_rates (pd.DataFrame): Time series of FX rates.
    rf_rates (pd.DataFrame): Time series of risk-free rates.
    transform_to_log_fx_rates (bool, default=True): If True, converts FX rates to log returns.
    transform_to_log_rf_rates (bool, default=True): If True, converts risk-free rates to log returns.
    rf_to_fx (dict, default=None): Mapping of risk-free rates to FX pairs.
    base_rf (str, default=None): Base risk-free rate to use for calculations.
    base_rf_series (pd.Series or pd.DataFrame, default=None): Time series of the base risk-free rate.
    periods_per_year (int or None, default=None): Factor for annualizing the returns.
    return_premium_series (bool, default=False): If True, returns the premium series instead of summary statistics.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    drop_before_keep (bool, default=False): If True, drops specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: Summary of the carry trade strategy statistics or premium series.
    """
    raise Exception("Function not available - needs testing prior to use")
    if periods_per_year is None:
        print(
            "Regression assumes 'periods_per_year' equals to 12 since it was not provided"
        )
        periods_per_year = 12

    fx_regressions = calc_fx_regression(
        fx_rates,
        rf_rates,
        transform_to_log_fx_rates,
        transform_to_log_rf_rates,
        rf_to_fx,
        base_rf,
        base_rf_series,
        periods_per_year,
    )

    fx_rates = fx_rates.copy()
    rf_rates = rf_rates.copy()
    if isinstance(base_rf_series, (pd.Series, pd.DataFrame)):
        base_rf_series = base_rf_series.copy()

    if rf_to_fx is None:
        rf_to_fx = {"GBP1M": "USUK", "EUR1M": "USEU", "CHF1M": "USSZ", "JPY1M": "USJP"}

    if transform_to_log_fx_rates:
        fx_rates = fx_rates.applymap(lambda x: math.log(x))

    if transform_to_log_rf_rates:
        rf_rates = rf_rates.applymap(lambda x: math.log(x + 1))

    if base_rf is None and base_rf_series is None:
        print(
            "No 'base_rf' or 'base_rf_series' was provided. Trying to use 'USD1M' as the base risk-free rate."
        )
        base_rf = "USD1M"
    if base_rf_series is None:
        base_rf_series = rf_rates[base_rf]

    all_expected_fx_premium = pd.DataFrame({})
    for rf in rf_to_fx.keys():
        try:
            rf_name = re.sub("[0-9]+M", "", rf)
        except:
            rf_name = rf
        fx_er_usd = (base_rf_series.shift(1) - rf_rates[rf].shift(1)).to_frame(
            "ER Over USD"
        )
        expected_fx_premium = (
            fx_regressions.loc[rf_name, "Alpha"]
            + (fx_regressions.loc[rf_name, "Base RF - Foreign RF Beta"] - 1) * fx_er_usd
        )
        expected_fx_premium = expected_fx_premium.rename(
            columns={"ER Over USD": rf_name}
        )
        all_expected_fx_premium = all_expected_fx_premium.join(
            expected_fx_premium, how="outer"
        )

    if return_premium_series:
        return _filter_columns_and_indexes(
            all_expected_fx_premium,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes,
            drop_before_keep=drop_before_keep,
        )

    all_expected_fx_premium = all_expected_fx_premium.dropna(axis=0)
    summary_statistics = (
        all_expected_fx_premium.applymap(lambda x: 1 if x > 0 else 0)
        .agg(["mean", "sum", "count"])
        .set_axis(
            [
                "% of Periods with Positive Premium",
                "Nº of Positive Premium Periods",
                "Total Number of Periods",
            ]
        )
    )
    summary_statistics = pd.concat(
        [
            summary_statistics,
            (
                all_expected_fx_premium.agg(
                    ["mean", "std", "min", "max", "skew", "kurtosis"]
                ).set_axis(["Mean", "Vol", "Min", "Max", "Skewness", "Kurtosis"])
            ),
        ]
    )
    summary_statistics = summary_statistics.transpose()
    summary_statistics["Annualized Mean"] = (
        summary_statistics["Mean"] * periods_per_year
    )
    summary_statistics["Annualized Vol"] = summary_statistics["Vol"] * math.sqrt(
        periods_per_year
    )

    return _filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes,
        drop_before_keep=drop_before_keep,
    )
