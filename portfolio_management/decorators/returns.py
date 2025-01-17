import inspect
import numpy as np
import pandas as pd
from functools import wraps

from portfolio_management.constants.columns import Columns
from portfolio_management.decorators.dates import validate_date_column


def validate_returns(check_dates=True, date_column=Columns.DATE.value):
    """
    Validates that a 'returns' parameter is a pd.DataFrame, pd.Series, or list.

    If the 'returns' parameter is a DataFrame, it ensures:
    - The 'returns' column contains numeric values with no NaNs or None.
    - The date column, if specified and validation is enabled, is sorted in ascending order.

    Parameters:
    check_dates (bool): Whether to validate the date column in a DataFrame. Default is True.
    date_column (str): The name of the date column to validate. Defaults to 'date'.

    Returns:
    function: A decorator to validate the returns and optionally the date column.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs).arguments
            returns = bound_arguments.get("returns")

            if returns is None:
                raise ValueError(
                    f"A 'returns' parameter must be provided as a pd.DataFrame, pd.Series, or list."
                )

            if not isinstance(returns, (pd.DataFrame, pd.Series, list)):
                raise ValueError(
                    f"The 'returns' parameter must be a pd.DataFrame, pd.Series, or list."
                )

            if isinstance(returns, pd.DataFrame):
                if Columns.RETURNS.value not in returns.columns:
                    raise KeyError(
                        f"The DataFrame must contain a '{Columns.RETURNS.value}' column."
                    )
                if not np.issubdtype(returns[Columns.RETURNS.value].dtype, np.number):
                    raise ValueError(
                        f"All elements in the '{Columns.RETURNS.value}' column of the DataFrame must be numeric."
                    )
                if returns[Columns.RETURNS.value].isna().any():
                    raise ValueError(
                        f"The '{Columns.RETURNS.value}' column in the DataFrame contains NaN or None values."
                    )

                if check_dates:
                    validate_date_column(date_column)(func)(*args, **kwargs)

            elif isinstance(returns, pd.Series):
                if not np.issubdtype(returns.dtype, np.number):
                    raise ValueError(
                        f"All elements in the '{Columns.RETURNS.value}' Series must be numeric."
                    )
                if returns.isna().any():
                    raise ValueError(
                        f"The '{Columns.RETURNS.value}' Series contains NaN or None values."
                    )

            elif isinstance(returns, list):
                if (not all(isinstance(x, (int, float)) for x in returns)) or any(
                    x is None or np.isnan(x) for x in returns
                ):
                    raise ValueError(
                        f"All elements in the '{Columns.RETURNS.value}' list must be numeric."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
