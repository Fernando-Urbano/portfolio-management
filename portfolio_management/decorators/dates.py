import inspect
import pandas as pd
from functools import wraps
from portfolio_management.constants.columns import Columns


def validate_date_column(column_name=Columns.DATE.value):
    """
    Validates that a DataFrame contains a valid date column with dates sorted in ascending order.

    Parameters:
    column_name (str): The name of the date column to validate. Defaults to 'date'.

    Returns:
    function: A decorator to validate the date column.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs).arguments
            dataframe = bound_arguments.get("returns")

            if dataframe is not None:
                if column_name not in dataframe.columns:
                    raise KeyError(
                        f"The DataFrame must contain a '{column_name}' column."
                    )
                if not pd.api.types.is_datetime64_any_dtype(dataframe[column_name]):
                    raise ValueError(
                        f"The '{column_name}' column must be of a datetime type."
                    )
                if not dataframe[column_name].is_monotonic_increasing:
                    raise ValueError(
                        f"The '{column_name}' column must be sorted in ascending order."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
