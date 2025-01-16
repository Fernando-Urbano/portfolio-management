# tests/test_risk.py

import pytest
import pandas as pd
import numpy as np
from portfolio_management.risk import (
    calc_ewma_volatility,
    calc_garch_volatility,
    calc_var_cvar_summary,
)
from portfolio_management.utils import create_returns_df
from tests.conftest import setup_pandas_display
from scipy.stats import norm
from arch import arch_model

@pytest.fixture
def test_returns():
    """
    Fixture to create a sample returns DataFrame for testing.
    """
    return create_returns_df(n_samples=500, n_assets=3, seed=42, variance_multiplier=.005)

@pytest.fixture
def test_excess_returns(test_returns):
    """
    Fixture to create excess returns by subtracting a risk-free rate.
    """
    rf = create_returns_df(n_samples=500, n_assets=1, avg_return=0.0002, seed=24)
    excess = test_returns.apply(lambda x: x - rf.iloc[:, 0].values)
    return excess

# -------------------------------
# Tests for calc_ewma_volatility
# -------------------------------

def test_calc_ewma_volatility_basic(test_returns):
    """
    Test basic functionality of calc_ewma_volatility.
    """
    asset = test_returns.columns[0]
    excess_returns = test_returns[asset]
    ewma_vol = calc_ewma_volatility(excess_returns)

    assert isinstance(ewma_vol, pd.Series), "EWMA volatility should return a pd.Series"
    assert len(ewma_vol) == len(excess_returns), "EWMA volatility series length mismatch"
    assert ewma_vol.index.equals(excess_returns.index), "EWMA volatility index mismatch"
    assert (ewma_vol >= 0).all(), "EWMA volatility should be non-negative"

def test_calc_ewma_volatility_custom_theta(test_returns):
    """
    Test calc_ewma_volatility with a custom theta.
    """
    asset = test_returns.columns[0]
    excess_returns = test_returns[asset]
    custom_theta = 0.9
    ewma_vol = calc_ewma_volatility(excess_returns, theta=custom_theta)

    assert isinstance(ewma_vol, pd.Series), "EWMA volatility should return a pd.Series"
    assert len(ewma_vol) == len(excess_returns), "EWMA volatility series length mismatch"

def test_calc_ewma_volatility_initial_vol(test_returns):
    """
    Test calc_ewma_volatility with a custom initial_vol.
    """
    asset = test_returns.columns[0]
    excess_returns = test_returns[asset]
    custom_initial_vol = 0.1
    ewma_vol = calc_ewma_volatility(excess_returns, initial_vol=custom_initial_vol, theta=.94)

    assert isinstance(ewma_vol, pd.Series), "EWMA volatility should return a pd.Series"
    assert abs(ewma_vol.iloc[0] - custom_initial_vol) / custom_initial_vol <= .1, "Initial volatility value mismatch"

def test_calc_ewma_volatility_empty_series():
    """
    Test calc_ewma_volatility with an empty series.
    """
    empty_series = pd.Series(dtype=float)
    ewma_vol = calc_ewma_volatility(empty_series)

    assert isinstance(ewma_vol, pd.Series), "EWMA volatility should return a pd.Series"
    assert ewma_vol.empty, "EWMA volatility should be empty for empty input"

def test_calc_ewma_volatility_single_data_point():
    """
    Test calc_ewma_volatility with a single data point.
    """
    single_point = pd.Series([0.01], index=[pd.Timestamp('2023-01-01')])
    ewma_vol = calc_ewma_volatility(single_point)

    assert isinstance(ewma_vol, pd.Series), "EWMA volatility should return a pd.Series"
    assert len(ewma_vol) == 1, "EWMA volatility series length mismatch"
    assert ewma_vol.iloc[0] == np.sqrt((0.01)**2 * (1 - 0.94) + (0.2 / np.sqrt(252))**2 * 0.94), "EWMA volatility calculation mismatch"

# -------------------------------
# Tests for calc_garch_volatility
# -------------------------------

def test_calc_garch_volatility_basic(test_excess_returns):
    """
    Test basic functionality of calc_garch_volatility.
    """
    asset = test_excess_returns.columns[0]
    excess_returns = test_excess_returns[asset]
    garch_vol = calc_garch_volatility(excess_returns)

    assert isinstance(garch_vol, pd.Series), "GARCH volatility should return a pd.Series"
    assert len(garch_vol) == len(excess_returns), "GARCH volatility series length mismatch"
    assert garch_vol.index.equals(excess_returns.index), "GARCH volatility index mismatch"
    assert (garch_vol >= 0).all(), "GARCH volatility should be non-negative"

def test_calc_garch_volatility_custom_order(test_excess_returns):
    """
    Test calc_garch_volatility with custom p and q.
    """
    asset = test_excess_returns.columns[0]
    excess_returns = test_excess_returns[asset]
    p, q = 2, 2
    garch_vol = calc_garch_volatility(excess_returns, p=p, q=q)

    assert isinstance(garch_vol, pd.Series), "GARCH volatility should return a pd.Series"
    assert len(garch_vol) == len(excess_returns), "GARCH volatility series length mismatch"

def test_calc_garch_volatility_empty_series():
    """
    Test calc_garch_volatility with an empty series.
    """
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        calc_garch_volatility(empty_series)

def test_calc_garch_volatility_constant_returns(test_excess_returns):
    """
    Test calc_garch_volatility with constant returns.
    """
    constant_returns = pd.Series(0.01, index=test_excess_returns.index)
    garch_vol = calc_garch_volatility(constant_returns)

    assert isinstance(garch_vol, pd.Series), "GARCH volatility should return a pd.Series"
    assert len(garch_vol) == len(constant_returns), "GARCH volatility series length mismatch"
    assert (garch_vol >= 0).all(), "GARCH volatility should be non-negative"

# -------------------------------
# Tests for calc_var_cvar_summary
# -------------------------------

def test_calc_var_cvar_summary_var_less_than_mean(test_returns):
    """
    Test that VaR is less than the mean return.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(returns)

    mean_return = summary["Returns"].mean()
    var = summary["Expanding 60 Historical VaR (5.00%)"].mean()

    assert var < mean_return, "VaR should be smaller than the mean return"

def test_calc_var_cvar_summary_cvar_less_than_var(test_returns):
    """
    Test that CVaR is less than VaR.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(returns)

    var = summary["Expanding 60 Historical VaR (5.00%)"].mean()
    cvar = summary["Expanding 60 Historical CVaR (5.00%)"].mean()

    assert cvar < var, "CVaR should be smaller than VaR"

def test_calc_var_cvar_summary_multiple_quantiles(test_returns):
    """
    Test calc_var_cvar_summary with multiple quantiles.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    quantiles = [0.01, 0.05, 0.1]

    for q in quantiles:
        summary = calc_var_cvar_summary(returns, quantile=q, window=60)
        assert f"Expanding 60 Historical VaR ({q:.2%})" in summary.columns
        assert f"Expanding 60 Historical CVaR ({q:.2%})" in summary.columns

def test_calc_var_cvar_summary_return_hit_ratio(test_returns):
    """
    Test calc_var_cvar_summary with return_hit_ratio=True.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(
        returns,
        return_hit_ratio=True,
        quantile=0.05,
        shift=1
    )

    assert isinstance(summary, pd.DataFrame), "Summary should return a pd.DataFrame"
    assert "Hit Ratio" in summary.columns, "Hit Ratio column missing"
    assert "Hit Ratio Error" in summary.columns, "Hit Ratio Error column missing"
    assert "Hit Ratio Absolute Error" in summary.columns, "Hit Ratio Absolute Error column missing"

def test_calc_var_cvar_summary_with_shift(test_returns):
    """
    Test calc_var_cvar_summary with shift parameter.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(returns, shift=2)

    # Since shift=2, the first two periods should have NaN for VaR and CVaR
    assert not summary["Expanding 60 Historical VaR (5.00%)"].isna().iloc[-1], "VaR should not be NaN after shift"

def test_calc_var_cvar_summary_with_full_time_sample(test_returns):
    """
    Test calc_var_cvar_summary with full_time_sample=True.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(returns, full_time_sample=True)

    # Should only contain expanding statistics
    for col in summary.columns:
        assert "expanding" in col.lower(), "Only expanding statistics should be present"

def test_calc_var_cvar_summary_with_drops(test_returns):
    """
    Test calc_var_cvar_summary with drop_columns and drop_indexes.
    """
    asset = test_returns.columns[0]
    returns = test_returns[[asset]]
    summary = calc_var_cvar_summary(
        returns,
        drop_columns=["Returns", "Expanding 60 Historical VaR (5.00%)"],
        drop_indexes=["Rolling 60 Historical VaR (5.00%)"],
        return_stats="Vol"
    )

    assert "Returns" not in summary.columns, "Returns should be dropped"
    assert "Expanding 60 Historical VaR (5.00%)" not in summary.columns, "Historical VaR should be dropped"
    assert "Rolling 60 Historical VaR (5.00%)" not in summary.columns, "Rolling VaR should be dropped"

def test_calc_var_cvar_summary_invalid_quantile(test_returns):
    """
    Test calc_var_cvar_summary with invalid quantile.
    """
    returns = test_returns[test_returns.columns[0]]
    with pytest.raises(ValueError):
        calc_var_cvar_summary(returns, quantile=1.5)

def test_calc_var_cvar_summary_zero_shift(test_returns):
    """
    Test calc_var_cvar_summary with shift=0.
    """
    returns = test_returns[test_returns.columns[0]]
    summary = calc_var_cvar_summary(returns, shift=0, return_hit_ratio=True)
    summary_shifted = calc_var_cvar_summary(returns, shift=1, return_hit_ratio=True)
    assert summary.equals(summary_shifted) == False, "Summaries with different shifts should not be exactly equal"

def test_calc_var_cvar_summary_negative_shift(test_returns):
    """
    Test calc_var_cvar_summary with negative shift (should handle gracefully or raise error).
    """
    returns = test_returns[test_returns.columns[0]]
    with pytest.raises(ValueError):
        calc_var_cvar_summary(returns, shift=-1)

def test_calc_var_cvar_summary_all_stats(test_returns):
    """
    Test calc_var_cvar_summary with return_stats='all'.
    """
    returns = test_returns[test_returns.columns[0]]
    summary = calc_var_cvar_summary(returns, return_stats="all", window=60)

    expected_columns = [
        "Returns",
        "Expanding 60 Historical VaR (5.00%)",
        "Rolling 60 Historical VaR (5.00%)",
        "Expanding 60 Volatility",
        "Rolling 60 Volatility",
        "EWMA 0.94 Volatility",
        "GARCH(1, 1) Volatility",
        "Expanding 60 Parametric VaR (5.00%)",
        "Rolling 60 Parametric VaR (5.00%)",
        "EWMA 0.94 Parametric VaR (5.00%)",
        "GARCH(1, 1) Parametric VaR (5.00%)",
        "Expanding 60 Historical CVaR (5.00%)",
        "Rolling 60 Historical CVaR (5.00%)",
        "Expanding 60 Parametrical CVaR (5.00%)",
        "Rolling 60 Parametrical CVaR (5.00%)",
        "EWMA 0.94 Parametrical CVaR (5.00%)",
        "GARCH(1, 1) Parametrical CVaR (5.00%)",
    ]

    for col in expected_columns:
        assert col in summary.columns, f"Missing expected column: {col}"

# -----------------------------------
# Additional Creative Tests
# -----------------------------------

def test_calc_var_cvar_summary_constant_returns():
    """
    Test calc_var_cvar_summary with constant returns.
    """
    constant_returns = pd.Series(0.01, index=pd.date_range(start='2023-01-01', periods=100), name='AAPL')
    summary = calc_var_cvar_summary(constant_returns)

    cvar = summary["Expanding 60 Historical CVaR (5.00%)"]
    cvar = cvar.dropna().tolist()
    assert cvar == [0.01] * len(cvar), "CVaR should be constant for constant returns"

def test_calc_var_cvar_summary_high_volatility(test_returns):
    """
    Test calc_var_cvar_summary with high volatility returns.
    """
    high_vol_returns = pd.Series(np.random.normal(0, 1, 500), index=pd.date_range(start='2023-01-01', periods=500))
    summary = calc_var_cvar_summary(high_vol_returns)
    summary = summary.dropna(axis=0)

    assert (summary["Expanding 60 Volatility"] > 0).all(), "Volatility should be positive"
    assert (summary["Rolling 60 Volatility"] > 0).all(), "Volatility should be positive"
    assert (summary["EWMA 0.94 Volatility"] > 0).all(), "EWMA Volatility should be positive"
    assert (summary["GARCH(1, 1) Volatility"] > 0).all(), "GARCH Volatility should be positive"

def test_calc_var_cvar_summary_invalid_inputs():
    """
    Test calc_var_cvar_summary with invalid inputs.
    """
    with pytest.raises(ValueError):
        # Non-numeric returns
        non_numeric_returns = pd.Series(["a", "b", "c"], index=pd.date_range(start='2023-01-01', periods=3))
        calc_var_cvar_summary(non_numeric_returns)

    with pytest.raises(ValueError):
        # Invalid window type
        returns = create_returns_df(n_samples=100, n_assets=1, seed=42)
        calc_var_cvar_summary(returns, window=-10)

def test_calc_var_cvar_summary_with_all_parameters(test_returns):
    """
    Test calc_var_cvar_summary with all parameters set.
    """
    asset = test_returns.columns[0]
    returns = test_returns[asset]
    summary = calc_var_cvar_summary(
        returns,
        quantile=0.05,
        window=60,
        return_hit_ratio=True,
        filter_first_hit_ratio_date='2000-06-01',
        full_time_sample=False,
        z_score=None,
        shift=1,
        normal_vol_formula=True,
        ewma_theta=0.95,
        ewma_initial_vol=0.15 / np.sqrt(252),
        garch_p=1,
        garch_q=1,
    )

    assert "Hit Ratio" in summary.columns, "Hit Ratio should be kept"
    assert all(abs(hr - .05) < .05 for hr in summary["Hit Ratio"].tolist())


if __name__ == "__main__":
    pytest.main([__file__])
