import pytest
import pandas as pd
import numpy as np
from portfolio_management.decorators.returns import validate_returns
from portfolio_management.constants.columns import Columns


@validate_returns()
def dummy_function(returns):
    return "Success"


def test_valid_dataframe():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            Columns.RETURNS.value: [0.1, 0.2, 0.3],
        }
    )
    assert dummy_function(returns=df) == "Success"


def test_missing_returns_column():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "other_column": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(
        KeyError,
        match=f"The DataFrame must contain a '{Columns.RETURNS.value}' column.",
    ):
        dummy_function(returns=df)


def test_non_numeric_returns():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            Columns.RETURNS.value: ["a", "b", "c"],
        }
    )
    with pytest.raises(
        ValueError,
        match=f"All elements in the '{Columns.RETURNS.value}' column of the DataFrame must be numeric.",
    ):
        dummy_function(returns=df)


def test_nan_in_returns():
    df_with_none = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            Columns.RETURNS.value: [0.1, None, 0.3],
        }
    )
    df_with_nan = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            Columns.RETURNS.value: [0.1, np.nan, 0.3],
        }
    )

    for df in [df_with_none, df_with_nan]:
        with pytest.raises(
            ValueError,
            match=f"The '{Columns.RETURNS.value}' column in the DataFrame contains NaN or None values.",
        ):
            dummy_function(returns=df)


def test_valid_series():
    series = pd.Series([0.1, 0.2, 0.3])
    assert dummy_function(returns=series) == "Success"


def test_nan_in_series():
    series_with_none = pd.Series([0.1, None, 0.3])
    series_with_nan = pd.Series([0.1, np.nan, 0.3])

    for series in [series_with_none, series_with_nan]:
        with pytest.raises(
            ValueError, match="The 'returns' Series contains NaN or None values."
        ):
            dummy_function(returns=series)


def test_valid_list():
    returns_list = [0.1, 0.2, 0.3]
    assert dummy_function(returns=returns_list) == "Success"


def test_non_numeric_list():
    returns_list = [0.1, "a", 0.3]
    returns_list_with_none = [0.1, None, 0.3]
    returns_list_with_nan = [0.1, np.nan, 0.3]

    for returns_list in [returns_list, returns_list_with_none, returns_list_with_nan]:
        with pytest.raises(
            ValueError, match="All elements in the 'returns' list must be numeric."
        ):
            dummy_function(returns=returns_list)


def test_missing_returns_argument():
    with pytest.raises(
        ValueError,
        match="A 'returns' parameter must be provided as a pd.DataFrame, pd.Series, or list.",
    ):
        dummy_function(returns=None)


def test_check_dates_enabled():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-03", "2023-01-01", "2023-01-02"]
            ),
            Columns.RETURNS.value: [0.1, 0.2, 0.3],
        }
    )

    @validate_returns(check_dates=True)
    def check_dates_function(returns):
        return "Success"

    with pytest.raises(
        ValueError,
        match=f"The '{Columns.DATE.value}' column must be sorted in ascending order.",
    ):
        check_dates_function(returns=df)


def test_check_dates_disabled():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-03", "2023-01-01", "2023-01-02"]
            ),
            Columns.RETURNS.value: [0.1, 0.2, 0.3],
        }
    )

    @validate_returns(check_dates=False)
    def no_date_check_function(returns):
        return "Success"

    assert no_date_check_function(returns=df) == "Success"


def test_returns_as_positional_argument():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            Columns.RETURNS.value: [0.1, 0.2, 0.3],
        }
    )

    @validate_returns()
    def positional_function(returns):
        return "Success"

    assert positional_function(df) == "Success"
