import pytest
import pandas as pd
import numpy as np
import re

import sys
import os
sys.path.append(os.pardir)
from portfolio_management.analysis import (
    calc_cross_section_regression,
    calc_regression,
    calc_iterative_regression,
)
from portfolio_management.utils import create_returns_df, create_rf_returns_df
from tests.conftest import setup_pandas_display  # Import the fixture

@pytest.fixture
def test_returns():
    return create_returns_df(n_samples=500, n_assets=10, seed=100)

@pytest.fixture
def test_factors():
    return create_returns_df(n_samples=500, n_assets=3, seed=200)

@pytest.fixture
def test_rf():
    return create_rf_returns_df(n_samples=500, seed=300)

@pytest.fixture
def test_regression_result(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    return calc_regression(y, X)

@pytest.fixture
def test_cross_section_regression(test_returns, test_factors, test_rf):
    return calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        return_model=False,
    )

# Tests for calc_regression
def test_calc_regression_basic_with_without_intercept(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    result = calc_regression(y, X)
    assert isinstance(result, pd.DataFrame)
    assert "Alpha" in result.columns.tolist()
    assert isinstance(result['Alpha'].iloc[0], float)
    result = calc_regression(y, X, intercept=False)
    assert "Alpha" in result.columns.tolist()
    assert isinstance(result['Alpha'].iloc[0], str)
    assert result['Alpha'].iloc[0] == "-"


def test_calc_regression_return_model(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    results = calc_regression(y, X, return_model=False)
    n_betas = sum([bool(re.search("Beta", c)) for c in results.columns.tolist()])
    assert n_betas == X.shape[1]
    results = calc_regression(y, X, return_model=True)
    len(results.params[1:]) == n_betas


def test_calc_regression_fitted_values(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    fitted = calc_regression(y, X, return_fitted_values=True)
    assert isinstance(fitted, pd.DataFrame)


def test_calc_regression_sortino_ratio(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    result = calc_regression(y, X, calc_sortino_ratio=True)
    assert "Sortino Ratio" in result.columns
    assert "Annualized Sortino Ratio" in result.columns


def test_calc_iterative_regression_basic(test_returns, test_factors):
    result = calc_iterative_regression(test_returns, test_factors)
    assert "Alpha" in result.columns
    for factor in test_factors.columns:
        assert f"{factor} Beta" in result.columns
    assert "R-Squared" in result.columns


def test_calc_iterative_regression_no_intercept(test_returns, test_factors):
    result = calc_iterative_regression(test_returns, test_factors, intercept=False)
    for factor in test_factors.columns:
        assert f"{factor} Beta" in result.columns
    assert all(alpha == "-" for alpha in result['Alpha'].tolist())


def test_calc_iterative_regression_with_treynor(test_returns, test_factors):
    result = calc_iterative_regression(test_returns, test_factors, calc_treynor_info_ratios=True)
    assert not "Treynor Ratio" in result.columns
    assert "Information Ratio" in result.columns
    result = calc_iterative_regression(test_returns, test_factors.iloc[:, 0], calc_treynor_info_ratios=True)
    assert "Treynor Ratio" in result.columns


def test_calc_iterative_regression_sortino(test_returns, test_factors):
    result = calc_iterative_regression(test_returns, test_factors, calc_sortino_ratio=True)
    assert "Sortino Ratio" in result.columns
    assert "Annualized Sortino Ratio" in result.columns


def test_calc_regression_with_timeframes(test_returns, test_factors):
    timeframes = {
        "first_half": (test_returns.index[0], test_returns.index[250]),
        "second_half": (test_returns.index[250], test_returns.index[-1]),
    }
    result = calc_regression(test_returns.iloc[:, 0], test_factors, timeframes=timeframes)
    assert isinstance(result, pd.DataFrame)
    assert any("first_half" in i[0] for i in result.index.tolist())
    assert any("second_half" in i[0] for i in result.index)


# Tests for calc_cross_section_regression
def test_calc_cross_section_regression_basic(test_returns, test_factors, test_rf):
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        return_model=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert "Eta" in result.columns


def test_calc_cross_section_regression_mae(test_returns, test_factors, test_rf):
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        return_mae=True,
    )
    assert "TS MAE" in result.columns
    assert "CS MAE" in result.columns

def test_calc_cross_section_regression_premiums(test_returns, test_factors, test_rf):
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        return_historical_premium=True,
        return_annualized_premium=True,
    )
    assert "Historical Premium" in result.columns or "Annualized Lambda" in result.columns


def test_calc_cross_section_regression_compare_premiums(test_returns, test_factors, test_rf):
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        compare_premiums=True,
    )
    # Assuming the output is filtered to only premiums comparison
    for col in result.columns:
        assert "Correlation" in col or "Lambda" in col or "Historical Premium" in col


def test_calc_cross_section_regression_filtering(test_returns, test_factors, test_rf):
    keep = ["Mean"]
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        keep_columns=keep,
    )
    for col in keep:
        assert col in result.columns
    assert len(result.columns) == len(keep)


def test_calc_cross_section_regression_invalid_rf_length(test_returns, test_factors, test_rf):
    rf_short = test_rf.iloc[:400]
    with pytest.raises(Exception, match='"rf" index must be the same lenght as "returns"'):
        calc_cross_section_regression(
            returns=test_returns,
            factors=test_factors,
            periods_per_year=12,
            provided_excess_returns=False,
            rf=rf_short,
            return_model=False,
        )


def test_calc_cross_section_regression_provided_excess_returns(test_returns, test_factors, test_rf):
    # Assuming returns are already excess
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=True,
        rf=test_rf,  # Should be ignored
        return_model=False,
    )
    assert "Fitted Mean" in result.columns



def test_calc_regression_empty_inputs():
    y = pd.Series(dtype=float)
    X = pd.DataFrame()
    with pytest.raises(Exception):
        calc_regression(y, X)


def test_calc_iterative_regression_empty_inputs():
    y = pd.DataFrame()
    X = pd.DataFrame()
    result = calc_iterative_regression(y, X)
    assert result.empty


def test_calc_cross_section_regression_no_returns(test_factors, test_rf):
    returns = pd.DataFrame()
    with pytest.raises(Exception):
        calc_cross_section_regression(
            returns=returns,
            factors=test_factors,
            periods_per_year=12,
            provided_excess_returns=False,
            rf=test_rf,
            return_model=False,
        )

def test_calc_regression_invalid_periods_per_year(test_returns, test_factors):
    y = test_returns.iloc[:, 0]
    X = test_factors
    with pytest.raises(ValueError, match="periods_per_year must be a positive integer"):
        calc_regression(y, X, periods_per_year=0)


def test_calc_iterative_regression_invalid_periods_per_year(test_returns, test_factors):
    with pytest.raises(ValueError, match="periods_per_year must be a positive integer"):
        calc_iterative_regression(test_returns, test_factors, periods_per_year=-1)


def test_calc_cross_section_regression_invalid_periods_per_year(test_returns, test_factors, test_rf):
    with pytest.raises(ValueError, match="periods_per_year must be a positive integer"):
        calc_cross_section_regression(
            returns=test_returns,
            factors=test_factors,
            periods_per_year=0,
            provided_excess_returns=False,
            rf=test_rf,
            return_model=False,
        )


# Additional Parameter Tests
def test_calc_regression_timeframes(test_returns, test_factors):
    timeframes = {
        "early": (test_returns.index[0], test_returns.index[250]),
        "late": (test_returns.index[250], test_returns.index[-1]),
    }
    result = calc_regression(
        y=test_returns.iloc[:, 0],
        X=test_factors,
        timeframes=timeframes
    )
    assert isinstance(result, pd.DataFrame)
    assert "Mean early" in result.columns or "Mean late" in result.columns


def test_calc_iterative_regression_with_drops(test_returns, test_factors):
    drop = list(test_factors.columns[:1])
    result = calc_iterative_regression(test_returns, test_factors, drop_columns=drop)
    for d in drop:
        assert d not in result.columns


def test_calc_cross_section_regression_with_drops(test_returns, test_factors, test_rf):
    drop_cols = list(test_factors.columns[:1])
    result = calc_cross_section_regression(
        returns=test_returns,
        factors=test_factors,
        periods_per_year=12,
        provided_excess_returns=False,
        rf=test_rf,
        drop_columns=drop_cols,
    )
    for d in drop_cols:
        assert d not in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
