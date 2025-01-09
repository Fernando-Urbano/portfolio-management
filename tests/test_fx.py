import pytest
import pandas as pd
import numpy as np
from portfolio_management.fx import calc_fx_exc_ret
from portfolio_management.fx import calc_fx_regression
from portfolio_management.fx import calc_dynamic_carry_trade
import math

import pandas.testing as pdt


def test_calc_fx_exc_ret():
    # Test basic functionality - Expected vs Actual Excess Returns with dummy data
    fx_rates = pd.DataFrame({'USEU': [0.93, 0.91, 0.9, 0.94, 0.98, 0.99, 0.97]},
                            index=pd.date_range("2023-01-01", periods=7))
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.011, 0.012, 0.013, 0.12, 0.11, 0.10],
        'EUR1M': [0.02, 0.019, 0.018, 0.022, 0.021, 0.023, 0.017]
    }, index=pd.date_range("2023-01-01", periods=7))

    # Manually calculate expected result
    expected_excess = pd.DataFrame({
        "EUR": [
            np.log(fx_rates['USEU'].iloc[1]) - np.log(fx_rates['USEU'].iloc[0]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[1]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[1])),
            np.log(fx_rates['USEU'].iloc[2]) - np.log(fx_rates['USEU'].iloc[1]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[2]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[2])),
            np.log(fx_rates['USEU'].iloc[3]) - np.log(fx_rates['USEU'].iloc[2]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[3]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[3])),
            np.log(fx_rates['USEU'].iloc[4]) - np.log(fx_rates['USEU'].iloc[3]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[4]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[4])),
            np.log(fx_rates['USEU'].iloc[5]) - np.log(fx_rates['USEU'].iloc[4]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[5]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[5])),
            np.log(fx_rates['USEU'].iloc[6]) - np.log(fx_rates['USEU'].iloc[5]) +
            (np.log(1 + rf_rates['EUR1M'].shift(1).iloc[6]) - np.log(1 + rf_rates['USD1M'].shift(1).iloc[6])),
        ],
    }, index=fx_rates.index[1:])

    # Calculate actual result
    actual_excess = calc_fx_exc_ret(
        fx_rates=fx_rates,
        rf_rates=rf_rates,
        transform_to_log_fx_rates=True,
        transform_to_log_rf_rates=True,
        rf_to_fx={'EUR1M': 'USEU'},  # Custom mapping
        base_rf='USD1M',
        return_exc_ret=True
    )

    pdt.assert_frame_equal(actual_excess, expected_excess)


    # Test  Mismatched Indices
    fx_rates = pd.DataFrame({'USEU': [0.93, 0.91, 0.9, 0.94]},
                            index=pd.date_range("2023-01-05", periods=4))
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.011, 0.012, 0.013],
        'EUR1M': [0.02, 0.019, 0.018, 0.022]
    }, index=pd.date_range("2023-01-01", periods=4))

    actual_mismatched = calc_fx_exc_ret(
        fx_rates=fx_rates,
        rf_rates=rf_rates,
        transform_to_log_fx_rates=True,
        transform_to_log_rf_rates=True,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
        return_exc_ret=True
    )

    pdt.assert_frame_equal(actual_mismatched, pd.DataFrame({"EUR":[]}))

    #Test data with NaN values
    fx_rates = pd.DataFrame({'USEU': [0.93, np.nan, 0.9, 0.94]},
                            index=pd.date_range("2023-01-01", periods=4))
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.011, 0.012, np.nan],
        'EUR1M': [0.02, np.nan, 0.018, 0.022]
    }, index=pd.date_range("2023-01-01", periods=4))

    actual_with_nan = calc_fx_exc_ret(
        fx_rates=fx_rates,
        rf_rates=rf_rates,
        transform_to_log_fx_rates=True,
        transform_to_log_rf_rates=True,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
        return_exc_ret=True
    )

    pdt.assert_frame_equal(actual_with_nan, pd.DataFrame({'EUR':[0.04939]}, index = actual_with_nan.index), check_exact = False,  atol = 1e-4)

def test_calc_fx_regression():
    fx_rates = pd.DataFrame({'USEU': [0.93, 0.91, 0.9, 0.94, 0.98, 0.99, 0.97]},
                            index=pd.date_range("2023-01-01", periods=7))
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.011, 0.012, 0.013, 0.12, 0.11, 0.10],
        'EUR1M': [0.02, 0.019, 0.018, 0.022, 0.021, 0.023, 0.017]
    }, index=pd.date_range("2023-01-01", periods=7))

    regression_stats = calc_fx_regression(
        fx_rates,
        rf_rates,
        annual_factor=12,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
    )

    assert list(regression_stats.columns) == ['Alpha', 'Annualized Alpha', 'R-Squared', 'Base RF - Foreign RF Beta', 'Treynor Ratio', 'Annualized Treynor Ratio', 'Information Ratio', 'Annualized Information Ratio', 'Tracking Error', 'Annualized Tracking Error', 'Fitted Mean', 'Annualized Fitted Mean']

    expected_shape = (1, 12)
    assert regression_stats.shape == expected_shape

    fx_rates = pd.DataFrame({'USEU': [0.93, 0.91, np.nan, 0.94, 0.98, 0.99, 0.97]},
                            index=pd.date_range("2023-01-01", periods=7))
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.011, 0.012, np.nan, 0.12, 0.11, 0.10],
        'EUR1M': [0.02, 0.019, 0.018, 0.022, np.nan, 0.023, 0.017]
    }, index=pd.date_range("2023-01-01", periods=7))

    regression_stats = calc_fx_regression(
        fx_rates,
        rf_rates,
        annual_factor=12,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
    )
    assert not regression_stats.isna().any().any()

    #Test for manual beta value
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.02, 0.03, 0.04],
        'EUR1M': [0, 0, 0, 0]
    }, index=pd.date_range("2023-01-01", periods=4))

    rate_differentials = np.log(1 + rf_rates['USD1M'])
    cumulative_differentials = rate_differentials.cumsum()

    fx_rates = pd.DataFrame({
        'USEU': np.exp(cumulative_differentials)
    }, index=rf_rates.index)

    regression_stats = calc_fx_regression(
        fx_rates,
        rf_rates,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
    )

    atol = 1e-5
    assert  abs(regression_stats['Base RF - Foreign RF Beta']['EUR'] - 1.0) <= atol
    assert abs(regression_stats['Alpha']['EUR'] - 0) <= atol

    #Test if works for multiple FX pairs
    fx_rates = pd.DataFrame({
        'USEU': [0.93, 0.91, 0.9, 0.94],
        'USUK': [1.3, 1.31, 1.33, 1.32]
    })
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.02, 0.03, 0.04],
        'EUR1M': [0, 0, 0, 0],
        'GBP1M': [0.03, 0.029, 0.028, 0.027]
    })

    regression_stats = calc_fx_regression(
        fx_rates,
        rf_rates,
        rf_to_fx={'EUR1M': 'USEU', 'GBP1M': 'USUK'},
        base_rf='USD1M',
    )
    expected_shape = (2, 12)
    assert regression_stats.shape == expected_shape

def test_calc_dynamic_carry_trade():

    # Test Basic Functionality for alpha = 0 and beta = 1 - Everything should be 0
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.02, 0.03, 0.04],
        'EUR1M': [0, 0, 0, 0]
    }, index=pd.date_range("2023-01-01", periods=4))

    rate_differentials = np.log(1 + rf_rates['USD1M'])
    cumulative_differentials = rate_differentials.cumsum()

    fx_rates = pd.DataFrame({
        'USEU': np.exp(cumulative_differentials)
    }, index=rf_rates.index)

    res = calc_dynamic_carry_trade(
        fx_rates=fx_rates,
        rf_rates=rf_rates,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
        return_premium_series=False  # Set True to get premium series instead of summary
    )

    atol = 1e-5
    assert abs(res['% of Periods with Positive Premium']['EUR'] - 0) <= atol
    assert abs(res['Nº of Positive Premium Periods']['EUR'] - 0) <= atol
    assert res['Total Number of Periods']['EUR'] == 3
    assert abs(res['Mean']['EUR'] - 0) <= atol
    assert abs(res['Vol']['EUR'] - 0) <= atol
    assert abs(res['Min']['EUR'] - 0) <= atol
    assert abs(res['Max']['EUR'] - 0) <= atol
    assert abs(res['Skewness']['EUR'] - 0) <= atol
    assert math.isnan(res['Kurtosis']['EUR']) == math.isnan(np.float64('nan'))
    assert abs(res['Annualized Mean']['EUR'] - 0) <= atol
    assert abs(res['Annualized Vol']['EUR'] - 0) <= atol

    #Test functionality with data that produce b = 2 and alpha = 0 (so we should have non-zero results)
    rf_rates = pd.DataFrame({
        'USD1M': [0.01, 0.02, 0.03, 0.04],  # Base risk-free rate
        'EUR1M': [0, 0, 0, 0]  # Foreign risk-free rate
    }, index=pd.date_range("2023-01-01", periods=4))

    rate_differentials = np.log(1 + rf_rates['USD1M'])[:-1]

    cumulative_differentials = rate_differentials.cumsum()
    fx_rates = pd.DataFrame({
        'USEU': np.exp(2 * cumulative_differentials)  # Adjusted for beta = 2
    }, index=rf_rates.index)

    res = calc_dynamic_carry_trade(
        fx_rates=fx_rates,
        rf_rates=rf_rates,
        rf_to_fx={'EUR1M': 'USEU'},
        base_rf='USD1M',
        return_premium_series = False
    )

    # Manually compute expected premium (since expected premium = alpha + (b - 1) rate diff then in our case since beta = 2 and alpha = 0, expected premium = rate diff
    expected_premiums = rate_differentials

    atol = 1e-5
    assert res['% of Periods with Positive Premium']['EUR'] == 1

    assert res['Nº of Positive Premium Periods']['EUR'] == 3

    assert res['Total Number of Periods']['EUR'] == 3

    expected_mean = expected_premiums.mean()
    assert abs(res['Mean']['EUR'] - expected_mean) <= atol

    expected_vol = expected_premiums.std()
    assert abs(res['Vol']['EUR'] - expected_vol) <= atol

    assert abs(res['Min']['EUR'] - expected_premiums.min()) <= atol
    assert abs(res['Max']['EUR'] - expected_premiums.max()) <= atol


    annual_factor = 12
    assert abs(res['Annualized Mean']['EUR'] - (expected_mean * annual_factor)) <= atol
    assert abs(res['Annualized Vol']['EUR'] - (expected_vol * np.sqrt(annual_factor))) <= atol

    assert list(res.columns) == ['% of Periods with Positive Premium', 'Nº of Positive Premium Periods',
       'Total Number of Periods', 'Mean', 'Vol', 'Min', 'Max', 'Skewness',
       'Kurtosis', 'Annualized Mean', 'Annualized Vol']
