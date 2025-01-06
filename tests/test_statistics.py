import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from portfolio_management.statistics import *
from tests.utils import create_returns_df


@pytest.fixture
def test_returns():
    return create_returns_df()

@pytest.fixture
def test_summary_statistics(test_returns):
    return calc_summary_statistics(test_returns)

# Test calc_negative_pct
def test_calc_negative_pct_positive(test_returns):
    result = calc_negative_pct(test_returns, calc_positive=True)
    assert isinstance(result, pd.DataFrame)
    assert "Nº Positive Returns" in result.index

def test_calc_negative_pct_negative(test_returns):
    result = calc_negative_pct(test_returns, calc_positive=False)
    assert isinstance(result, pd.DataFrame)
    assert "% Negative Returns" in result.index

def test_calc_negative_pct_with_drops(test_returns):
    drop = list(test_returns.columns[0:1])
    result = calc_negative_pct(test_returns, drop_columns=drop)
    for d in drop:
        assert d not in result.columns

def test_calc_negative_pct_series(test_returns):
    series = test_returns.iloc[:, 0]
    result = calc_negative_pct(series)
    assert isinstance(result, pd.DataFrame)

# Test get_best_and_worst
def test_get_best_and_worst(test_summary_statistics):
    for stat in test_summary_statistics.columns:
        result = get_best_and_worst(test_summary_statistics, stat=stat)
        assert isinstance(result, pd.DataFrame)
        assert result.index[0] == test_summary_statistics[stat].idxmax()
        assert result.index[-1] == test_summary_statistics[stat].idxmin()

def test_get_best_and_worst_invalid_stat(test_summary_statistics):
    with pytest.raises(Exception, match=r"not in \"summary_statistics\""):
        get_best_and_worst(test_summary_statistics, stat="Nonexistent Stat")

# Test calc_correlations
def test_calc_correlations(test_returns):
    result = calc_correlations(test_returns, return_heatmap=False)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (test_returns.shape[1], test_returns.shape[1])  # Square matrix

def test_calc_correlations_heatmap(test_returns):
    result = calc_correlations(test_returns, return_heatmap=True)
    assert result is not None  # Verifying heatmap is generated

def test_calc_correlations_with_col_drops(test_returns):
    drop = list(test_returns.columns[0:1])
    result = calc_correlations(test_returns, drop_columns=drop, return_heatmap=False)
    for d in drop:
        assert d not in result.columns
        assert d in result.index

def test_calc_correlations_with_idx_drops(test_returns):
    drop = list(test_returns.columns[0:1])
    result = calc_correlations(test_returns, drop_indexes=drop, return_heatmap=False)
    for d in drop:
        assert d in result.columns
        assert d not in result.index

def test_calc_correlations_with_idx_col_drops(test_returns):
    drop = list(test_returns.columns[0:1])
    result = calc_correlations(test_returns, drop_columns=drop, drop_indexes=drop, return_heatmap=False)
    for d in drop:
        assert d not in result.columns
        assert d not in result.index

# Test calc_summary_statistics
def test_calc_summary_statistics(test_returns):
    result = calc_summary_statistics(test_returns)
    assert isinstance(result, pd.DataFrame)
    assert "Mean" in result.columns
    assert "Annualized Mean" in result.columns

def test_calc_summary_statistics_with_drops(test_returns):
    drop = list(test_returns.columns[0:1])
    result = calc_summary_statistics(test_returns, drop_indexes=drop)
    for d in drop:
        assert d not in result.index

if __name__ == "__main__":
    pytest.main()
