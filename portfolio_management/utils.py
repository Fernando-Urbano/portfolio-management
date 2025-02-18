import pandas as pd
import numpy as np
import datetime
from typing import Literal, Union
import warnings

from sklearn.datasets import make_sparse_spd_matrix

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")

PERIODS_PER_YEAR_MAP = {
    "D": 360,
    "DU": 252,
    "W": 52,
    "BM": 12,
    "ME": 12,
    "BQ": 4,
    "BA": 2,
    "A": 1,
}

PERIODS_PER_YEAR = 12


def read_excel_default(
    excel_name: str,
    index_col: int = 0,
    parse_dates: bool = True,
    print_sheets: bool = False,
    sheet_name: str = None,
    **kwargs,
):
    """
    Read an Excel file into a pandas DataFrame with default options.

    This function reads the specified Excel file and returns a DataFrame using the provided
    options. If `print_sheets` is True, it prints the names and first few rows of all sheets and
    returns None. The function also sets the index name to "date" if the index appears to contain
    date-like values.

    Parameters
    ----------
    excel_name : str
        Path to the Excel file.
    index_col : int, optional
        Column to use as the row labels of the DataFrame. Defaults to 0.
    parse_dates : bool, optional
        Whether to parse dates in the index. Defaults to True.
    print_sheets : bool, optional
        If True, prints the sheet names and the first few rows of each sheet and returns None.
        Defaults to False.
    sheet_name : str or int, optional
        Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs
        Additional keyword arguments passed to `pd.read_excel`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the specified Excel sheet. If `print_sheets` is True,
        the function returns None.

    Examples
    --------
    >>> df = read_excel_default("data.xlsx", index_col=0, parse_dates=True)
    >>> df.head()
    """
    if print_sheets:
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f"Sheet {n}:")
                print(", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print("\n" * 2)
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    returns = pd.read_excel(
        excel_name,
        index_col=index_col,
        parse_dates=parse_dates,
        sheet_name=sheet_name,
        **kwargs,
    )
    if returns.index.name is not None:
        if returns.index.name.lower() in ["date", "dates"]:
            returns.index.name = "date"
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = "date"
    return returns


def _filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str],
    drop_before_keep: bool = False,
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    df = df.copy()
    # Columns
    if keep_columns is not None:
        keep_columns = (
            "(?i)" + "|".join(keep_columns)
            if isinstance(keep_columns, list)
            else "(?i)" + keep_columns
        )
    else:
        keep_columns = None
    if drop_columns is not None:
        drop_columns = (
            "(?i)" + "|".join(drop_columns)
            if isinstance(drop_columns, list)
            else "(?i)" + drop_columns
        )
    else:
        drop_columns = None
    if not drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    if drop_columns is not None:
        df = df.drop(columns=df.filter(regex=drop_columns).columns)
    if drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    # Indexes
    if keep_indexes is not None:
        keep_indexes = (
            "(?i)" + "|".join(keep_indexes)
            if isinstance(keep_indexes, list)
            else "(?i)" + keep_indexes
        )
    else:
        keep_indexes = None
    if drop_indexes is not None:
        drop_indexes = (
            "(?i)" + "|".join(drop_indexes)
            if isinstance(drop_indexes, list)
            else "(?i)" + drop_indexes
        )
    else:
        drop_indexes = None
    if not drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    if drop_indexes is not None:
        df = df.drop(index=df.filter(regex=drop_indexes, axis=0).index)
    if drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    return df


def _transform_periods_per_year(freq) -> int:
    return PERIODS_PER_YEAR_MAP.get(freq)


def _calc_periods_per_year(dates) -> str:
    """
    Given a list/array-like of dates, attempt to infer the frequency
    (daily, weekly, monthly, etc.) and return one of the keys in
    PERIODS_PER_YEAR_MAP:
        "D"  -> 252   (Daily)
        "W"  -> 52    (Weekly)
        "BM" -> 12    (Monthly, not necessarily month-end)
        "ME" -> 12    (Monthly, all month-end)
        "BQ" -> 4     (Quarterly)
        "BA" -> 2     (Semiannual, or even annual in practice)

    If fewer than 20 dates are provided, the function raises a ValueError
    forcing the user to specify the frequency manually.

    :param dates: A list/array-like of datetime objects (or date strings).
    :return: A string key from PERIODS_PER_YEAR_MAP indicating the inferred frequency.
    """

    if len(dates) < 20:
        raise ValueError(
            "Not enough data points to auto-detect 'periods_per_year'. "
            "At least 20 dates are required. Please specify manually."
        )

    dates = pd.to_datetime(dates)
    dates = np.sort(dates)

    day_diffs = np.diff(dates) / np.timedelta64(1, "D")  # length n-1
    median_gap = np.median(day_diffs)

    # ----- Frequency detection thresholds -----
    #
    # Typical day-gap heuristics (approximate):
    #
    #   < 2 days   => "D"  (Daily)
    #   < 10 days  => "W"  (Weekly)
    #   < 40 days  => "ME" or "BM" (Monthly)
    #   < 80 days  => "BQ" (Quarterly)
    #   < 200 days => "BA" (Semiannual)
    #   >= 200 days=> "A" (also used for ~annual in this map)
    #
    if median_gap < 2:
        max_gap = np.max(day_diffs)
        frequency = "DU" if max_gap > 2 else "D"
    elif median_gap < 10:
        frequency = "W"
    elif median_gap < 40:
        # Distinguish "month-end" vs. "business-monthly"
        is_month_end = all(d == (d + pd.tseries.offsets.MonthEnd(0)) for d in dates)
        frequency = "ME" if is_month_end else "BM"
    elif median_gap < 80:
        frequency = "BQ"
    elif median_gap < 200:
        frequency = "BA"
    else:
        frequency = "A"
    return _transform_periods_per_year(frequency)


def create_returns_df(
    n_samples: int = 1000,
    n_assets: int = 5,
    avg_return: float = 0.004,
    alpha_sparsity: float = 0.3,
    seed: int = 42,
    end_date: str = "2024-01-01",
    date_frequecy: Union[Literal["ME", "BM", "BQ", "BA", "W", "D"]] = "ME",  # For month
    variance_multiplier: float = 0.03,
    truncate: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic returns DataFrame.

    This function creates a DataFrame of synthetic asset returns using a multivariate normal
    distribution. It generates a random covariance matrix via a sparse positive-definite matrix,
    scales it by `variance_multiplier`, and simulates returns for a specified number of assets
    and samples. Optionally, returns below -1 are truncated to -0.95.

    Parameters
    ----------
    n_samples : int, optional
        Number of time periods (samples) to generate. Defaults to 1000.
    n_assets : int, optional
        Number of assets (columns) to generate. Defaults to 5.
    avg_return : float, optional
        Average return for each asset. Defaults to 0.004.
    alpha_sparsity : float, optional
        Sparsity parameter for generating the covariance matrix. Defaults to 0.3.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    end_date : str, optional
        End date for the date range index. Defaults to "2024-01-01".
    date_frequecy : {"ME", "BM", "BQ", "BA", "W", "D"}, optional
        Frequency for the date range. Defaults to "ME" (month-end).
    variance_multiplier : float, optional
        Multiplier for scaling the covariance matrix. Must be between 0 and 0.5. Defaults to 0.03.
    truncate : bool, optional
        If True, truncates returns less than -1 to -0.95. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame of synthetic returns with a datetime index and asset columns.

    Raises
    ------
    ValueError
        If `variance_multiplier` is not between 0 and 0.5.

    Examples
    --------
    >>> df_returns = create_returns_df(n_samples=500, n_assets=3, avg_return=0.005)
    >>> df_returns.head()
    """
    if variance_multiplier > 0.5 or variance_multiplier <= 0:
        raise ValueError("variance_multiplier must be between 0 and 0.5")
    rng = np.random.RandomState(seed)
    asset_names = [
        "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))
        for i in range(n_assets)
    ]
    cov_matrix = make_sparse_spd_matrix(n_dim=n_assets, alpha=alpha_sparsity)
    cov_matrix /= np.max(cov_matrix) / variance_multiplier
    returns = np.random.multivariate_normal(
        np.ones(n_assets) * avg_return, cov_matrix, n_samples
    )
    if truncate:
        returns[returns < -1] = -0.95
    returns_df = pd.DataFrame(returns, columns=asset_names)
    returns_df.index = pd.date_range(
        end=end_date, periods=n_samples, freq=date_frequecy
    )
    return returns_df


def create_rf_returns_df(
    n_samples: int = 1000,
    avg_rf_rate: float = 0.002,
    ts_auto_correlation: float = 0.8,
    seed: int = 42,
    std_rf_rate: float = 0.01,
    end_date: str = "2024-01-01",
    date_frequecy: Union[Literal["ME", "BM", "BQ", "BA", "W", "D"]] = "ME",
) -> pd.DataFrame:
    """
    Generate a synthetic risk-free returns DataFrame.

    This function creates a DataFrame of synthetic risk-free returns using a normal distribution.
    It applies a time-series autocorrelation model to simulate persistence in the risk-free rate.
    The resulting DataFrame has a datetime index and a single column labeled "RF".

    Parameters
    ----------
    n_samples : int, optional
        Number of time periods (samples) to generate. Defaults to 1000.
    avg_rf_rate : float, optional
        Average risk-free rate. Defaults to 0.002.
    ts_auto_correlation : float, optional
        Autocorrelation coefficient for the risk-free rate time series. Defaults to 0.8.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    std_rf_rate : float, optional
        Standard deviation of the risk-free rate. Defaults to 0.01.
    end_date : str, optional
        End date for the date range index. Defaults to "2024-01-01".
    date_frequecy : {"ME", "BM", "BQ", "BA", "W", "D"}, optional
        Frequency for the date range. Defaults to "ME" (month-end).

    Returns
    -------
    pd.DataFrame
        DataFrame of synthetic risk-free returns with a datetime index and a single column "RF".

    Examples
    --------
    >>> rf_df = create_rf_returns_df(n_samples=500)
    >>> rf_df.head()
    """
    rng = np.random.RandomState(seed)
    rf_returns = rng.normal(avg_rf_rate, std_rf_rate, n_samples)
    rf_returns = pd.Series(rf_returns)
    for i in range(1, n_samples):
        rf_returns[i] = (
            rf_returns[i] * (1 - ts_auto_correlation)
            + rf_returns[i - 1] * ts_auto_correlation
        )
    rf_returns.index = pd.date_range(
        end=end_date, periods=n_samples, freq=date_frequecy
    )
    return rf_returns.to_frame("RF")


def define_periods_per_year(returns, periods_per_year=None):
    """
    Determine the number of periods per year for a returns DataFrame.

    This function validates and returns the number of periods per year. If `periods_per_year`
    is provided as a float, it is converted to an integer (if whole). If not provided, the
    function attempts to infer the frequency from the DataFrame's date information using
    internal heuristics.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame containing returns data. The function expects either a "date" column or a
        DatetimeIndex.
    periods_per_year : int or float, optional
        Number of periods per year. Must be a positive integer. If None, the frequency is inferred
        from the returns data.

    Returns
    -------
    int
        The number of periods per year.

    Raises
    ------
    ValueError
        If `periods_per_year` is not a positive integer or if the date frequency cannot be inferred.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"returns": [0.01, 0.02, 0.03]},
    ...                   index=pd.date_range("2021-01-01", periods=3))
    >>> define_periods_per_year(df)
    252  # Example output for daily data
    """
    if isinstance(periods_per_year, float):
        if periods_per_year % 1 != 0:
            raise ValueError("periods_per_year must be a positive integer.")
        else:
            periods_per_year = int(periods_per_year)
    if isinstance(periods_per_year, int):
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer.")
        return periods_per_year
    elif periods_per_year is None:
        if isinstance(returns, pd.DataFrame):
            if "date" in [c.lower() for c in returns.columns]:
                dates = returns["date"]
            elif isinstance(returns.index, pd.DatetimeIndex):
                dates = returns.index
            else:
                raise ValueError(
                    "Could not infer dates from the DataFrame. Either specify manually or ensure the DataFrame has a 'date' column or a DatetimeIndex."
                )
        return _calc_periods_per_year(dates)
    else:
        raise ValueError("periods_per_year must be a positive integer or None.")


def clean_returns_df(returns):
    """
    Clean and validate the returns data.

    This function accepts returns data in various formats (a list of Series,
    a pandas Series, or a pandas DataFrame) and performs several cleaning steps
    to ensure consistency. The cleaning process includes:

    - Merging multiple pandas Series (if provided as a list) into a single DataFrame.
    - Converting a pandas Series to a DataFrame and renaming its column if necessary.
    - Renaming any column that contains "date" (case insensitive) to "date" and setting
      it as the index.
    - Ensuring that the index is a DatetimeIndex or PeriodIndex, converting it if needed.
    - Converting all data values to floats.
    - Dropping rows with missing values and printing a warning if any rows are removed.

    Parameters
    ----------
    returns : list, pd.Series, or pd.DataFrame
        The input returns data. It can be provided as:
        - A list of pandas Series objects to be merged into a DataFrame.
        - A single pandas Series, which will be converted into a DataFrame.
        - A pandas DataFrame containing the returns data.
        In all cases, the index should represent dates or be convertible to datetime.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with:
        - A DatetimeIndex (or PeriodIndex) named "date".
        - All values converted to float.
        - Rows with missing values removed.

    Raises
    ------
    ValueError
        - If the input DataFrame or Series is empty.
        - If the index contains invalid datetime values or cannot be parsed.
        - If the conversion of data values to floats fails.

    Examples
    --------
    >>> import pandas as pd
    >>> from portfolio_management.utils import clean_returns_df
    >>> # Example with a Series:
    >>> dates = pd.date_range("2023-01-01", periods=3)
    >>> returns_series = pd.Series([0.1, 0.2, 0.3], index=dates)
    >>> clean_df = clean_returns_df(returns_series)
    >>> clean_df.head()
                0
    date
    2023-01-01  0.1
    2023-01-02  0.2
    2023-01-03  0.3
    """
    if isinstance(returns, list):
        returns_list = returns[:]
        returns = pd.DataFrame({})
        for series in returns_list:
            returns = returns.merge(
                series, right_index=True, left_index=True, how="outer"
            )
    elif isinstance(returns, (pd.Series, pd.DataFrame)):
        if returns.empty:
            raise ValueError("The DataFrame is empty.")
        returns = returns.copy()

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
        if returns.columns[0] == 0:
            returns.columns = ["RT"]

    if "date" in returns.columns.str.lower():
        returns = returns.rename({"Date": "date"}, axis=1)
        returns = returns.set_index("date")

    returns.index.name = "date"
    if not isinstance(returns.index, pd.DatetimeIndex) and not isinstance(
        returns.index, pd.PeriodIndex
    ):
        try:
            returns.index = pd.to_datetime(returns.index, errors="coerce")
            if returns.index.isnull().any():
                raise ValueError(
                    "Index contains invalid datetime values. Ensure the 'returns' index is fully parsable."
                )
        except Exception as e:
            raise ValueError(f"Failed to process the 'date' index: {e}")

    try:
        returns = returns.apply(lambda x: x.astype(float))
    except Exception as e:
        raise ValueError(f"Failed to convert the DataFrame to floats: {e}")
    prev_len_index = returns.apply(lambda x: len(x))
    returns = returns.dropna(axis=0)
    new_len_index = returns.apply(lambda x: len(x))
    if not (prev_len_index == new_len_index).all():
        print("Some columns had NaN values and were dropped")

    return returns
