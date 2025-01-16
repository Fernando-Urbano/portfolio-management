import pytest
import pandas as pd
from portfolio_management.decorators.dates import validate_date_column
from portfolio_management.constants.columns import Columns


@validate_date_column()
def dummy_function(returns):
    return "Success"


def test_valid_date_column():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "other_column": [1, 2, 3],
        }
    )
    assert dummy_function(returns=df) == "Success"


def test_missing_date_column():
    df = pd.DataFrame({"other_column": [1, 2, 3]})
    with pytest.raises(
        KeyError, match=f"The DataFrame must contain a '{Columns.DATE.value}' column."
    ):
        dummy_function(returns=df)


def test_incorrect_date_type():
    df = pd.DataFrame(
        {
            Columns.DATE.value: [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ],  # Strings, not datetime
            "other_column": [1, 2, 3],
        }
    )
    with pytest.raises(
        ValueError,
        match=f"The '{Columns.DATE.value}' column must be of a datetime type.",
    ):
        dummy_function(returns=df)


def test_unsorted_dates():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-03", "2023-01-01", "2023-01-02"]
            ),
            "other_column": [1, 2, 3],
        }
    )
    with pytest.raises(
        ValueError,
        match=f"The '{Columns.DATE.value}' column must be sorted in ascending order.",
    ):
        dummy_function(returns=df)


def test_custom_column_name():
    @validate_date_column(column_name="custom_date")
    def custom_dummy_function(returns):
        return "Success"

    df = pd.DataFrame(
        {
            "custom_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "other_column": [1, 2, 3],
        }
    )
    assert custom_dummy_function(returns=df) == "Success"


def test_custom_column_name_missing():
    @validate_date_column(column_name="custom_date")
    def custom_dummy_function(returns):
        return "Success"

    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "other_column": [1, 2, 3],
        }
    )
    with pytest.raises(
        KeyError, match="The DataFrame must contain a 'custom_date' column."
    ):
        custom_dummy_function(returns=df)


def test_returns_as_positional_argument():
    df = pd.DataFrame(
        {
            Columns.DATE.value: pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "other_column": [1, 2, 3],
        }
    )

    @validate_date_column()
    def positional_function(returns):
        return "Success"

    assert positional_function(df) == "Success"
