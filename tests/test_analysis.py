import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.pardir)
from portfolio_management.analysis import *
from tests.utils import *


# calc_cross_section_regression depends on both calc_regression, which depends on calc_iterative regression.
# Consequently, testing this function will also test the other two functions in the analysis module.
def test_calc_cross_section_regression():
    # Mock data
    returns = create_returns_df()

    factors = create_returns_df(n_assets=2, seed=43)

    rf = create_rf_returns_df(seed=44)

    # Execute function
    result = calc_cross_section_regression(
        returns=returns,
        factors=factors,
        annual_factor=12,
        provided_excess_returns=False,
        rf=rf,
        return_model=False,
        name="TestModel",
        return_mae=True,
        intercept_cross_section=True,
        return_historical_premium=True,
        return_annualized_premium=True,
        compare_premiums=False
    )

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame."
    assert f"{factors.columns[0]} Lambda" in result.columns, f"{factors.columns[0]} Lambda should be in result."
    assert f"{factors.columns[1]} Lambda" in result.columns, f"{factors.columns[1]} Lambda should be in result."
    assert not result.empty, "Result should not be empty."

    # Check premium columns
    assert f"{factors.columns[0]} Annualized Lambda" in result.columns, f"Annualized premium for {factors.columns[0]} is missing."
    assert f"{factors.columns[1]} Historical Premium" in result.columns, f"Historical premium for {factors.columns[1]} is missing."
    assert "CS MAE" in result.columns, "CS MAE should be in result."
    assert "TS MAE" in result.columns, "TS MAE should be in result."

    # Verify data integrity
    assert result.iloc[0][f"{factors.columns[0]} Lambda"] != 0, f"{factors.columns[0]} Lambda should not be zero."
    assert result.iloc[0][f"{factors.columns[1]} Historical Premium"] > 0, "Historical Premium should be positive."

if __name__ == "__main__":
    pytest.main()
